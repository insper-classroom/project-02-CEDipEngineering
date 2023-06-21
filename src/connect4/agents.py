from typing import Optional, List, Sequence, Union, Dict, Tuple, Any, Callable

from pettingzoo.classic import connect_four_v3
import gymnasium as gym
from torch import nn
import numpy as np
import torch

from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, PPOPolicy
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.data import to_torch_as

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
    agent = DQNPolicy(
        model=net,
        optim=optim,
        discount_factor=0.999,
        estimation_step=3,
        target_update_freq=240,
    )
    return agent

def make_ppo_agent(hidden_sizes: Optional[List[int]] = [256, 128, 128, 128, 128, 256], softmax = False) -> BasePolicy:
    
    # define policy
    def dist(p):
        return torch.distributions.Categorical(logits=p)
    
    env = PettingZooEnv(connect_four_v3.env())
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )
    input_size = np.prod(observation_space.shape or observation_space["observation"].shape or observation_space["observation"].n)
    action_size = env.action_space.shape or env.action_space.n
    net = PPONet(
        input_shape=observation_space.shape or observation_space["observation"].shape or observation_space["observation"].n,
        action_shape=env.action_space.shape or env.action_space.n,
        softmax=softmax,
    )
    
    actor = Actor(net, action_size, softmax_output=False)
    critic = Critic(net)
    optim = torch.optim.NAdam(
        ActorCritic(actor, critic).parameters(), lr=1e-5, eps=1e-5
    )

    agent = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=0.99,
        action_space=env.action_space,
    )
    return agent

class PPONet(nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int],
        action_shape: Sequence[int],
        softmax = False,
    ) -> None:
        super().__init__()
        layers = [
            torch.nn.Flatten(),
            torch.nn.Linear(np.prod(input_shape), 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256,256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256,256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, action_shape),
        ] 
        if softmax:
            layers.append(torch.nn.Softmax())
        self.net = torch.nn.Sequential(*layers)
        # output_dim attribute needed for tianshou Actor __init__()
        self.output_dim = action_shape

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        mask = getattr(obs, "mask", None)
        obs = torch.as_tensor(obs["obs"], dtype=torch.float32)
        logits = self.net(obs)
        # print(mask, to_torch_as(mask, logits))
        # if mask is not None:
        #     logits *= to_torch_as(mask, logits)
        return logits, state