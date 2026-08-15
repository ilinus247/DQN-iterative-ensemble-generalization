"""
Independent DQN (IQL) for VMAS's "simple_tag" (predator-prey) using TorchRL.

VMAS (Vectorized Multi-Agent Simulator) runs many copies of the environment
in parallel as batched PyTorch tensors, optionally on GPU -- much faster than
PettingZoo's CPU-only simple_tag for the same underlying scenario.

Two teams of agents:
  - "adversary": several predators, chasing the prey (share one Q-network)
  - "agent":      one (or more) prey, evading the predators (share one Q-network)

Each team is trained as an independent DQN learner (IQL): its own Q-network,
target network, optimizer, and replay buffer, treating the other team as
part of the environment. No communication, no centralized critic.

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

# ---- Environment ----
max_steps = 100  # episode length (VMAS scenario horizon)
n_chasers = 3
n_evaders = 2
n_obstacles = 2

# Desired total frames per training batch; VMAS runs this many parallel
# copies of the env so that num_vmas_envs * max_steps ~= frames_per_batch.
frames_per_batch = 6_000
num_vmas_envs = max(1, frames_per_batch // max_steps)


def make_env(num_envs, seed):
    base_env = VmasEnv(
        scenario="simple_tag",
        num_envs=num_envs,
        continuous_actions=False,  # discrete actions -> DQN applies
        max_steps=max_steps,
        device=device,
        seed=seed,
        # Scenario-specific kwargs
        num_good_agents=n_evaders,
        num_adversaries=n_chasers,
        num_landmarks=n_obstacles,
    )
    return TransformedEnv(base_env)


env = make_env(num_vmas_envs, seed=1)

# Separate, smaller vectorized env for evaluation -- never reuse the training
# env, since the collector keeps it stepping continuously between batches and
# an out-of-band reset/rollout would corrupt that state. VMAS lets us run
# several eval episodes in parallel instead of looping.
n_eval_envs = 50
eval_env = make_env(n_eval_envs, seed=1)

# group_map groups agents that share an action/observation spec -- for
# simple_tag this naturally splits into {"adversary": [...], "agent": [...]}.
group_map = env.group_map
print("Groups:", group_map)

# ---- Per-team Q-networks, policies, exploration, losses, and buffers ----
policies = {}
policies_explore = {}
loss_modules = {}
target_updaters = {}
optimizers = {}
exploration_modules = {}
replay_buffers = {}

for group, agents in group_map.items():
    n_agents = len(agents)
    obs_spec = env.observation_spec[group, "observation"]
    action_spec = env.full_action_spec[group, "action"]
    n_obs = obs_spec.shape[-1]
    n_actions = action_spec.space.n  # Categorical spec: shape[-1] is n_agents, not n_actions

    # share_params=True == one shared Q-network for every agent on this team
    # (parameter sharing, the standard MARL trick for homogeneous agents).
    net = MultiAgentMLP(
        n_agent_inputs=n_obs,
        n_agent_outputs=n_actions,
        n_agents=n_agents,
        centralised=False,  # each agent only sees its own observation
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

    exploration_module = EGreedyModule(
        spec=action_spec,
        annealing_num_steps=500_000,  # in frames; VMAS racks up frames fast
        eps_init=1.0,
        eps_end=0.05,
        action_key=(group, "action"),
    )
    policy_explore = TensorDictSequential(policy, exploration_module)

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
    loss_module.make_value_estimator(gamma=0.99)

    policies[group] = policy
    policies_explore[group] = policy_explore
    loss_modules[group] = loss_module
    target_updaters[group] = SoftUpdate(loss_module, eps=0.999)
    optimizers[group] = Adam(loss_module.parameters(), lr=5e-4)
    exploration_modules[group] = exploration_module
    # Each team gets its own replay buffer, holding only that team's own
    # transitions -- no cross-group data, no shared-key collisions.
    replay_buffers[group] = ReplayBuffer(storage=LazyTensorStorage(max_size=1_000_000))

# One combined policy that acts for every team in a single forward pass --
# this is what the collector and eval rollouts actually call.
full_policy = TensorDictSequential(*policies.values())
full_policy_explore = TensorDictSequential(*policies_explore.values())

# ---- Data collection ----
total_frames = -1  # collect indefinitely; we break out manually
max_frames_safety_cap = 20_000_000  # VMAS collects frames very fast

collector = Collector(
    env,
    full_policy_explore,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    device=device,
)

# ---- Solve criteria ----
# There's no universal "solved" score for simple_tag (asymmetric, densely
# shaped reward, not a fixed target like CartPole's 475/500). Instead of an
# arbitrary absolute threshold, we detect a *plateau*: track a reference
# rolling average, and stop once the rolling average hasn't moved by more
# than `min_change` (in either direction) for `patience` consecutive checks.
eval_window = 20  # smooth over more evals -- single-digit windows are noisy
patience = 40  # how many consecutive rolling-avg checks must stay within min_change of each other
min_change = 10.0  # max allowed spread (max - min) across the last `patience` rolling averages
recent_returns = deque(maxlen=eval_window)
rolling_avg_history = deque(maxlen=patience)


# The env only reports a single, shared done/terminated per env (episode
# termination is global across all agents in simple_tag), but each group's
# reward has an extra per-agent dimension. This pulls out just one group's
# subtree from a freshly collected batch, with done/terminated broadcast to
# match that group's per-agent shape -- done once per collected batch here,
# rather than repeatedly on every sampled minibatch.
def extract_group_transition(data, group):
    group_td = data.select(group, ("next", group))
    reward_shape = group_td["next", group, "reward"].shape
    next_done = data["next", "done"]
    next_terminated = (
        data["next", "terminated"] if ("next", "terminated") in data.keys(True) else next_done
    )
    group_td.set(
        ("next", group, "done"), next_done.unsqueeze(-2).expand(*reward_shape).clone()
    )
    group_td.set(
        ("next", group, "terminated"),
        next_terminated.unsqueeze(-2).expand(*reward_shape).clone(),
    )
    return group_td


# ---- Checkpointing ----
checkpoint_path = "simple_tag_dqn_checkpoint.pt"
checkpoint_every = 20  # iterations between periodic saves during training


def save_checkpoint(path):
    checkpoint = {
        "group_map": group_map,
        "config": {
            "n_obs": {
                group: env.observation_spec[group, "observation"].shape[-1]
                for group in group_map
            },
            "n_actions": {
                group: env.full_action_spec[group, "action"].space.n for group in group_map
            },
            "n_agents": {group: len(agents) for group, agents in group_map.items()},
            "num_good_agents": n_evaders,
            "num_adversaries": n_chasers,
            "num_landmarks": n_obstacles,
            "max_steps": max_steps,
        },
        "policy_state_dicts": {group: policies[group].state_dict() for group in group_map},
    }
    torch.save(checkpoint, path)


# ---- Logging ----
writer = SummaryWriter(log_dir="runs/simple_tag_iql")

# ---- Training loop ----
batch_size = 128
updates_per_batch = 100
total_frames_collected = 0

for i, data in enumerate(collector):
    total_frames_collected += data.numel()

    for group in group_map:
        replay_buffers[group].extend(extract_group_transition(data, group).reshape(-1))

    if any(len(replay_buffers[group]) < batch_size for group in group_map):
        continue

    for _ in range(updates_per_batch):
        for group in group_map:
            sample = replay_buffers[group].sample(batch_size).to(device)

            loss_vals = loss_modules[group](sample)
            loss = loss_vals["loss"]

            optimizers[group].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(loss_modules[group].parameters(), max_norm=10.0)
            optimizers[group].step()

            target_updaters[group].step()

    for group in group_map:
        exploration_modules[group].step(data.numel())

    # ---- Evaluation ----
    # eval_env runs n_eval_envs episodes in parallel in one rollout call.
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        eval_env.reset()
        eval_rollout = eval_env.rollout(max_steps, full_policy)
    # Sum reward over time, mean over agents on the team, mean over the
    # parallel eval envs. Only tracking the adversary team since simple_tag's
    # reward is zero-sum here -- the agent (prey) team's return is just the
    # negative of this, so it carries no extra information.
    adversary_return = (
        eval_rollout["next", "adversary", "reward"].sum(1).mean().item()
    )
    recent_returns.append(adversary_return)
    rolling_avg = sum(recent_returns) / len(recent_returns)

    writer.add_scalar("adversary/return", adversary_return, i)
    writer.add_scalar("adversary/rolling_avg", rolling_avg, i)

    print(
        f"iter {i:4d} | frames {total_frames_collected:,} | "
        f"adversary return {adversary_return:.2f} | "
        f"rolling avg ({len(recent_returns)}) {rolling_avg:.2f}"
    )

    if i % checkpoint_every == 0:
        save_checkpoint(f"train/simple_tag_dqn_checkpoint_iter{i}.pt")
        print(f"  (checkpoint saved to {checkpoint_path} at iter {i})")

    rolling_avg_history.append(rolling_avg)

    if len(rolling_avg_history) == patience:
        spread = max(rolling_avg_history) - min(rolling_avg_history)
        if spread <= min_change:
            print(
                f"Plateaued: the last {patience} rolling averages all stayed "
                f"within {min_change} of each other (spread={spread:.2f}), "
                f"after {total_frames_collected} frames (iter {i})."
            )
            break

    if total_frames_collected >= max_frames_safety_cap:
        print(
            f"Hit safety cap of {max_frames_safety_cap} frames without solving. "
            f"Best rolling avg was {rolling_avg:.2f}."
        )
        break

collector.shutdown()
writer.close()

# ---- Save final trained policies ----
save_checkpoint(checkpoint_path)
print(f"Saved final trained policies to {checkpoint_path}")