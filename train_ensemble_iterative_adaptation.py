"""
Build an ensemble of "agent" (prey) policies via iterated best-response /
opponent adaptation, starting from an existing IQL checkpoint.

Algorithm:
    1. Load initial Ag (agent/prey), Ad (adversary) from an existing
       checkpoint (produced by dqn_torchrl_marl_vmas_simple_tag.py).
    2. For each of N rounds:
       a. Freeze Ad, train ONLY Ag for K iterations  -> Ag'
       b. Freeze Ag', train ONLY Ad for K iterations -> Ad'
       c. Save this round's Ad' as one ensemble member.
    3. Ad and Ag both continue evolving cumulatively round-to-round (this is
       one continuous alternating adaptation trajectory, not N independent
       runs) -- the N saved Ad' snapshots capture the adversary's policy at
       different points along that trajectory, each having faced a
       differently-adapted Ag.

Install deps:
    pip install torchrl vmas tensordict torch
"""

from collections import deque

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch.utils.tensorboard import SummaryWriter
from torchrl.envs import TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import EGreedyModule, MultiAgentMLP, QValueModule
from torchrl.collectors import Collector
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.objectives import DQNLoss, SoftUpdate
from torch.optim import Adam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Hyperparameters ----
initial_checkpoint_path = "simple_tag_dqn_checkpoint.pt"  # from the original IQL training script
ensemble_out_dir = "ensemble"
N = 4  # number of rounds -> number of ensemble members saved
K = 30  # collector iterations per phase (per Ad-only or Ag-only training step)

frames_per_batch = 6_000  # same convention as the original training script
batch_size = 128
updates_per_batch = 100
lr = 5e-4
gamma = 0.99
target_update_eps = 0.999
grad_clip_norm = 10.0
eps_init = 1.0
eps_end = 0.05
n_eval_envs = 50  # for the periodic printed eval only, not the ensemble itself

import os

os.makedirs(ensemble_out_dir, exist_ok=True)

# ---- Load the initial checkpoint (env/scenario config comes from here) ----
initial_checkpoint = torch.load(initial_checkpoint_path, map_location=device)
group_map = initial_checkpoint["group_map"]
config = initial_checkpoint["config"]

max_steps = config["max_steps"]
num_vmas_envs = max(1, frames_per_batch // max_steps)


def make_env(num_envs, seed):
    base_env = VmasEnv(
        scenario="simple_tag",
        num_envs=num_envs,
        continuous_actions=False,
        max_steps=max_steps,
        device=device,
        seed=seed,
        num_good_agents=config["num_good_agents"],
        num_adversaries=config["num_adversaries"],
        num_landmarks=config["num_landmarks"],
    )
    return TransformedEnv(base_env)


env = make_env(num_vmas_envs, seed=0)
eval_env = make_env(n_eval_envs, seed=1)

# ---- Logging ----
# One writer for the whole run. global_step increments once per collector
# iteration across ALL phases/rounds (not reset per phase), so each group's
# curve is continuous and comparable across the full adaptation trajectory
# even though the two groups' phases are interleaved and each group's own
# curve has gaps (steps) during the other group's phases.
writer = SummaryWriter(log_dir="runs/simple_tag_ensemble_adaptation")
global_step = [0]


def build_group(group):
    """(Re)build all per-group training components, loading this group's
    weights from the initial checkpoint. Returns a dict of components."""
    n_agents = config["n_agents"][group]
    n_obs = config["n_obs"][group]
    n_actions = config["n_actions"][group]
    action_spec = env.full_action_spec[group, "action"]

    net = MultiAgentMLP(
        n_agent_inputs=n_obs,
        n_agent_outputs=n_actions,
        n_agents=n_agents,
        centralised=False,
        share_params=True,
        device=device,
        depth=2,
        num_cells=128,
    )
    value_module = TensorDictModule(
        net,
        in_keys=[(group, "observation")],
        out_keys=[(group, "action_value")],
    )
    q_module = QValueModule(
        action_space="categorical",
        spec=action_spec,
        action_value_key=(group, "action_value"),
        out_keys=[(group, "action"), (group, "action_value"), (group, "chosen_action_value")],
    )
    policy = TensorDictSequential(value_module, q_module)
    # Load this group's trained weights BEFORE building the loss module, so
    # DQNLoss's internal (delay_value=True) target-network copy starts from
    # the loaded weights too, not from a fresh random init.
    policy.load_state_dict(initial_checkpoint["policy_state_dicts"][group])

    loss_module = DQNLoss(
        value_network=policy,
        action_space="categorical",
        delay_value=True,
        double_dqn=True,
    )
    loss_module.set_keys(
        reward=(group, "reward"),
        action=(group, "action"),
        action_value=(group, "action_value"),
        value=(group, "chosen_action_value"),
        done=(group, "done"),
        terminated=(group, "terminated"),
    )
    loss_module.make_value_estimator(gamma=gamma)

    return {
        "policy": policy,
        "action_spec": action_spec,
        "loss_module": loss_module,
        "target_updater": SoftUpdate(loss_module, eps=target_update_eps),
        "optimizer": Adam(loss_module.parameters(), lr=lr),
        "replay_buffer": ReplayBuffer(storage=LazyTensorStorage(max_size=1_000_000)),
    }


components = {group: build_group(group) for group in group_map}


def make_phase_exploration(group):
    """Fresh EGreedyModule for a group entering its active phase -- a new
    decay curve sized to K collector iterations, not a continuation of the
    original (already fully decayed) training-run schedule."""
    return EGreedyModule(
        spec=components[group]["action_spec"],
        annealing_num_steps=max(1, K * frames_per_batch // 2),
        eps_init=eps_init,
        eps_end=eps_end,
        action_key=(group, "action"),
    )


def extract_group_transition(data, group):
    group_td = data.select(group, ("next", group))
    reward_shape = group_td["next", group, "reward"].shape
    next_done = data["next", "done"]
    next_terminated = (
        data["next", "terminated"] if ("next", "terminated") in data.keys(True) else next_done
    )
    group_td.set(("next", group, "done"), next_done.unsqueeze(-2).expand(*reward_shape).clone())
    group_td.set(
        ("next", group, "terminated"),
        next_terminated.unsqueeze(-2).expand(*reward_shape).clone(),
    )
    return group_td


def run_phase(active_group, frozen_group, phase_label):
    """Train `active_group` for K collector iterations while `frozen_group`
    acts greedily (no exploration, no updates)."""
    print(f"--- {phase_label}: training '{active_group}', '{frozen_group}' frozen ---")

    active = components[active_group]
    frozen = components[frozen_group]

    # Fresh exploration schedule for the active group this phase; frozen
    # group has no exploration module at all this phase -> acts greedily.
    exploration_module = make_phase_exploration(active_group)
    active_policy_explore = TensorDictSequential(active["policy"], exploration_module)
    phase_policy = TensorDictSequential(active_policy_explore, frozen["policy"])

    # Stale vs. the just-changed opponent -- clear before this phase's data.
    active["replay_buffer"] = ReplayBuffer(storage=LazyTensorStorage(max_size=1_000_000))

    collector = Collector(
        env,
        phase_policy,
        frames_per_batch=frames_per_batch,
        total_frames=-1,
        device=device,
    )

    recent_returns = deque(maxlen=10)
    for k, data in enumerate(collector):
        active["replay_buffer"].extend(extract_group_transition(data, active_group).reshape(-1))

        if len(active["replay_buffer"]) >= batch_size:
            for _ in range(updates_per_batch):
                sample = active["replay_buffer"].sample(batch_size).to(device)
                loss_vals = active["loss_module"](sample)
                loss = loss_vals["loss"]

                active["optimizer"].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    active["loss_module"].parameters(), max_norm=grad_clip_norm
                )
                active["optimizer"].step()
                active["target_updater"].step()

        exploration_module.step(data.numel())

        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            eval_env.reset()
            eval_rollout = eval_env.rollout(max_steps, phase_policy)
        ret = eval_rollout["next", active_group, "reward"].sum(1).mean().item()
        recent_returns.append(ret)
        writer.add_scalar(f"{active_group}/{phase_label}/return", ret, k)
        print(
            f"  [{phase_label}] k={k+1}/{K} | {active_group} return {ret:.2f} | "
            f"avg({len(recent_returns)}) {sum(recent_returns) / len(recent_returns):.2f}"
        )

        if k + 1 >= K:
            break

    collector.shutdown()


def save_ensemble_member(round_idx):
    path = f"{ensemble_out_dir}/adversary_ensemble_member_{round_idx}.pt"
    checkpoint = {
        "group_map": group_map,
        "config": config,
        "policy_state_dict": components["adversary"]["policy"].state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"Saved ensemble member {round_idx} to {path}")


# ---- Main alternating adaptation loop ----
for round_idx in range(N):
    print(f"\n===== Round {round_idx + 1}/{N} =====")
    run_phase(active_group="agent", frozen_group="adversary", phase_label=f"round{round_idx}-phaseA")
    run_phase(active_group="adversary", frozen_group="agent", phase_label=f"round{round_idx}-phaseB")
    save_ensemble_member(round_idx)

print(f"\nDone. {N} ensemble members saved to {ensemble_out_dir}/")
