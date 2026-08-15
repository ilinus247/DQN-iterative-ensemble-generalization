"""
Render a trained IQL DQN policy on VMAS's simple_tag.

Loads the checkpoint saved by dqn_torchrl_marl_vmas_simple_tag.py, rebuilds
the same per-team networks, runs a greedy (no-exploration) episode, and
saves it as a GIF -- works headless, no display needed.

Install deps (same as training, plus imageio for the GIF):
    pip install torchrl vmas tensordict torch imageio

Run:
    python render_simple_tag.py
"""

import imageio
import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MultiAgentMLP, QValueModule

# ---- Settings ----
checkpoint_path = "simple_tag_dqn_checkpoint.pt"
num_episodes = 5
out_path = "simple_tag_rollout_rac.gif"
fps = 15


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


device = torch.device("cpu")  # rendering a single env -- CPU is plenty

checkpoint = torch.load(checkpoint_path, map_location=device)
group_map = checkpoint["group_map"]
config = checkpoint["config"]

env = TransformedEnv(
    VmasEnv(
        scenario="simple_tag",
        num_envs=1,
        continuous_actions=False,
        max_steps=config["max_steps"],
        device=device,
        seed=1,
        num_good_agents=config["num_good_agents"],
        num_adversaries=config["num_adversaries"],
        num_landmarks=config["num_landmarks"],
        respawn_at_catch=True
    )
)

policies = {}
for group in group_map:
    action_spec = env.full_action_spec[group, "action"]
    policy = build_policy(
        group,
        config["n_obs"][group],
        config["n_actions"][group],
        config["n_agents"][group],
        action_spec,
        device,
    )
    policy.load_state_dict(checkpoint["policy_state_dicts"][group])
    policy.eval()
    policies[group] = policy

full_policy = TensorDictSequential(*policies.values())

frames = []


def rendering_callback(env, _tensordict):
    frames.append(env.render(mode="rgb_array"))


for ep in range(num_episodes):
    env.reset()
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        rollout = env.rollout(
            max_steps=config["max_steps"],
            policy=full_policy,
            callback=rendering_callback,
            auto_cast_to_device=True,
        )

    reward_str = " | ".join(
        f"{group} reward {rollout['next', group, 'reward'].sum().item():.2f}"
        for group in group_map
    )
    print(f"episode {ep+1}: {reward_str}")

imageio.mimsave(out_path, frames, fps=fps)
print(f"Saved {len(frames)} frames across {num_episodes} episode(s) to {out_path}")
