"""
Opponent generalization test for the trained IQL simple_tag adversaries.

The adversaries co-adapted to one specific prey policy during training --
that's not the same as "learned to hunt". This script pits the FINAL
adversary policy against two prey opponents:

  1. "trained (co-adapted)": the prey's final checkpoint -- same opponent
     the adversaries actually trained against. This is the baseline/
     in-distribution condition.
  2. "random prey": prey takes uniformly random actions -- a maximally
     out-of-distribution opponent.

A big adversary-return drop from (1) to (2) suggests the adversaries
overfit to their specific training partner's behavior rather than learning
a general pursuit strategy.

Install deps (same as training):
    pip install torchrl vmas tensordict torch
"""

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch.utils.tensorboard import SummaryWriter
from torchrl.envs import TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MultiAgentMLP, QValueModule

# ---- Settings ----
adversary_checkpoint_path = "simple_tag_dqn_checkpoint.pt"  # final checkpoint
n_eval_envs = 100  # parallel episodes per condition
device = torch.device("cpu")

writer = SummaryWriter(log_dir="runs/simple_tag_opponent_generalization")


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
    """Fills in a group's action with uniformly random samples from its
    action spec -- a maximally out-of-distribution opponent baseline."""

    def __init__(self, group, action_spec):
        self.group = group
        self.action_spec = action_spec

    def __call__(self, tensordict):
        tensordict.set((self.group, "action"), self.action_spec.rand())
        return tensordict


# ---- Load the adversary (the policy we're actually testing) ----
adversary_checkpoint = torch.load(adversary_checkpoint_path, map_location=device)
group_map = adversary_checkpoint["group_map"]
config = adversary_checkpoint["config"]

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


# Build against the NORMAL (training-matching) env first, since we need its
# action specs to construct the policies below.
env = make_eval_env(respawn_at_catch=False)

adversary_action_spec = env.full_action_spec["adversary", "action"]
adversary_policy = build_policy(
    "adversary",
    config["n_obs"]["adversary"],
    config["n_actions"]["adversary"],
    config["n_agents"]["adversary"],
    adversary_action_spec,
    device,
)
adversary_policy.load_state_dict(adversary_checkpoint["policy_state_dicts"]["adversary"])
adversary_policy.eval()

prey_action_spec = env.full_action_spec["agent", "action"]
prey_policy = build_policy(
    "agent",
    config["n_obs"]["agent"],
    config["n_actions"]["agent"],
    config["n_agents"]["agent"],
    prey_action_spec,
    device,
)
prey_policy.load_state_dict(adversary_checkpoint["policy_state_dicts"]["agent"])
prey_policy.eval()

conditions = {
    "trained": prey_policy,
    "random prey": RandomGroupPolicy("agent", prey_action_spec),
}

print(f"{'Condition':<25s} {'Dynamics':<20s} {'Adversary return':>18s}")
print("-" * 65)
mean_returns = []
labels = []
for respawn_at_catch in [False, True]:
    eval_env = env if not respawn_at_catch else make_eval_env(respawn_at_catch=True)
    dynamics_label = "respawn_at_catch" if respawn_at_catch else "normal"

    for name, prey_policy in conditions.items():

        def full_policy(tensordict):
            tensordict = adversary_policy(tensordict)
            tensordict = prey_policy(tensordict)
            return tensordict

        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            eval_env.reset()
            rollout = eval_env.rollout(config["max_steps"], full_policy)

        # Per-env (i.e. per-episode) returns -- logged as their own
        # histogram per condition, in addition to the aggregate-means one
        # below.
        per_episode_returns = rollout["next", "adversary", "reward"].sum(1).squeeze(-1)
        adversary_return = per_episode_returns.mean().item()
        print(f"{name:<25s} {dynamics_label:<20s} {adversary_return:>18.2f}")

        tag = f"{name}/{dynamics_label}".replace(" ", "_")
        writer.add_histogram(f"adversary_return/{tag}", per_episode_returns, global_step=0)

        mean_returns.append(adversary_return)
        labels.append(f"{name} / {dynamics_label}")

writer.close()