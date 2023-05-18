from typing import Optional, List

from pettingzoo.classic import connect_four_v3
import gymnasium as gym
import torch

from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy
from tianshou.utils.net.common import Net

def make_dqn_agent(hidden_sizes: Optional[List[int]] = [256, 128, 128, 128, 128, 256]) -> BasePolicy:
    env = PettingZooEnv(connect_four_v3.env())
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )
    # model
    net = Net(
        state_shape=observation_space.shape or observation_space["observation"].shape
        or observation_space["observation"].n,
        action_shape=env.action_space.shape or env.action_space.n,
        hidden_sizes=hidden_sizes,
    )
    optim = torch.optim.NAdam(net.parameters(), lr=1e-5)
    agent_learn = DQNPolicy(
        model=net,
        optim=optim,
        discount_factor=0.999,
        estimation_step=3,
        target_update_freq=240,
    )
    return agent_learn