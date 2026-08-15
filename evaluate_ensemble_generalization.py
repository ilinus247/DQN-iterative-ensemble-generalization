"""
Compare the ensemble "agent" (prey) policy against the original single
agent policy on generalization conditions, testing the hypothesis that the
ensemble (built via iterated opponent adaptation -- see
train_ensemble_iterative_adaptation.py) generalizes to OOD conditions better
than the original single policy.

Two policies under test:
  - "single (original)": the agent policy from the original IQL checkpoint.
  - "ensemble":           uniformly-random member selection across the N
                           saved ensemble snapshots.

Conditions varied (independent of which agent policy is under test):
  - adversary opponent: the ORIGINAL trained adversary, vs. a UNIFORMLY
    RANDOM adversary (maximally OOD).
  - dynamics: normal (matches training) vs. respawn_at_catch=True (harder,
    never seen during training).

That's a 2 (agent policy) x 2 (opponent) x 2 (dynamics) = 8-condition grid.

Install deps:
    pip install torchrl vmas tensordict torch
"""

import glob

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch.utils.tensorboard import SummaryWriter
from torchrl.envs import TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MultiAgentMLP, QValueModule

# ---- Settings ----
initial_checkpoint_path = "simple_tag_dqn_checkpoint.pt"
ensemble_dir = "ensemble"
ensemble_glob = f"{ensemble_dir}/adversary_ensemble_member_*.pt"
n_eval_envs = 100
device = torch.device("cpu")

writer = SummaryWriter(log_dir="runs/simple_tag_ensemble_generalization")


def build_policy(group, n_obs, n_actions, n_agents, action_spec, device):
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
    return TensorDictSequential(value_module, q_module)


class RandomGroupPolicy:
    def __init__(self, group, action_spec):
        self.group = group
        self.action_spec = action_spec

    def __call__(self, tensordict):
        tensordict.set((self.group, "action"), self.action_spec.rand())
        return tensordict


class EnsemblePolicy:
    """Each call, uniformly picks ONE of the N member policies and uses it
    to produce this step's action for the whole batch (see module docstring
    for the per-step vs. per-env granularity note)."""

    def __init__(self, members):
        self.members = members

    def __call__(self, tensordict):
        member = self.members[torch.randint(len(self.members), (1,)).item()]
        return member(tensordict)


# ---- Load checkpoints ----
initial_checkpoint = torch.load(initial_checkpoint_path, map_location=device)
group_map = initial_checkpoint["group_map"]
config = initial_checkpoint["config"]


def make_eval_env(respawn_at_catch):
    return TransformedEnv(
        VmasEnv(
            scenario="simple_tag",
            num_envs=n_eval_envs,
            continuous_actions=False,
            max_steps=config["max_steps"],
            device=device,
            seed=0,
            num_good_agents=config["num_good_agents"],
            num_adversaries=config["num_adversaries"],
            num_landmarks=config["num_landmarks"],
            respawn_at_catch=respawn_at_catch,
        )
    )


env = make_eval_env(respawn_at_catch=False)

adversary_action_spec = env.full_action_spec["adversary", "action"]
agent_action_spec = env.full_action_spec["agent", "action"]


def load_policy(group, action_spec, state_dict):
    policy = build_policy(
        group,
        config["n_obs"][group],
        config["n_actions"][group],
        config["n_agents"][group],
        action_spec,
        device,
    )
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


# Original single agent policy (the baseline we're trying to beat).
single_adversary_policy = load_policy(
    "adversary", adversary_action_spec, initial_checkpoint["policy_state_dicts"]["adversary"]
)

# Ensemble: load every saved member.
member_paths = sorted(glob.glob(ensemble_glob))
if not member_paths:
    raise FileNotFoundError(
        f"No ensemble members found matching {ensemble_glob} -- run "
        "train_ensemble_iterative_adaptation.py first."
    )
ensemble_members = []
for path in member_paths:
    ckpt = torch.load(path, map_location=device)
    ensemble_members.append(load_policy("adversary", adversary_action_spec, ckpt["policy_state_dict"]))
ensemble_policy = EnsemblePolicy(ensemble_members)
print(f"Loaded {len(ensemble_members)} ensemble members from {ensemble_dir}/")

# Original trained adversary (in-distribution opponent condition).
original_agent_policy = load_policy(
    "agent", agent_action_spec, initial_checkpoint["policy_state_dicts"]["agent"]
)

adversary_conditions = {
    "single (original)": single_adversary_policy,
    "ensemble": ensemble_policy,
}
agent_conditions = {
    "original agent": original_agent_policy,
    "random agent": RandomGroupPolicy("agent", agent_action_spec),
}

print(
    f"{'Adversary':<20s} {'Agent policy':<22s}  {'Dynamics':<20s} "
    f"{'Adversary return':>15s}"
)
print("-" * 80)
results = []
for respawn_at_catch in [False, True]:
    eval_env = env if not respawn_at_catch else make_eval_env(respawn_at_catch=True)
    dynamics_label = "respawn_at_catch" if respawn_at_catch else "normal"

    for adv_name, adversary_policy in adversary_conditions.items():
        for agent_name, agent_policy in agent_conditions.items():

            def full_policy(tensordict):
                tensordict = adversary_policy(tensordict)
                tensordict = agent_policy(tensordict)
                return tensordict

            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                eval_env.reset()
                rollout = eval_env.rollout(config["max_steps"], full_policy)

            per_episode_returns = rollout["next", "adversary", "reward"].sum(1).squeeze(-1)
            adv_return = per_episode_returns.mean().item()
            print(
                f"{adv_name:<22s} {agent_name:<20s} {dynamics_label:<20s} "
                f"{adv_return:>15.2f}"
            )

            tag = f"{agent_name}/{adv_name}/{dynamics_label}".replace(" ", "_")
            writer.add_histogram(f"adv_return_eval/{tag}", per_episode_returns, global_step=0)
            results.append((adv_name, agent_name, dynamics_label, adv_return))

writer.close()

print()
print("Ensemble vs. single, same condition, side by side:")
by_key = {(a, b, c): v for a, b, c, v in results}
for agent_name in agent_conditions:
    for dynamics_label in ["normal", "respawn_at_catch"]:
        single_v = by_key[("single (original)", agent_name, dynamics_label)]
        ensemble_v = by_key[("ensemble", agent_name, dynamics_label)]
        delta = ensemble_v - single_v
        print(
            f"  [{agent_name} / {dynamics_label}] single={single_v:.2f}  "
            f"ensemble={ensemble_v:.2f}  delta={delta:+.2f}"
        )
