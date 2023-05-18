"""
This is a minimal example of using Tianshou with MARL to train agents.

Author: Will (https://github.com/WillDudley)

Python version used: 3.8.10

Modified by: CEDip (https://github.com/CEDipEngineering)
"""

import os
from typing import Optional, Tuple
from collections import deque

from resnet18 import ResNet, BasicBlock

import gymnasium as gym
import numpy as np
import torch

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy, PPOPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.discrete import Actor, Critic, IntrinsicCuriosityModule


from pettingzoo.classic import go_v5, tictactoe_v3


def _get_agents(
        agent_learn: Optional[BasePolicy] = None,
        agent_opponent: Optional[BasePolicy] = None,
        optim: Optional[torch.optim.Optimizer] = None,
    ) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = _get_env()

    if agent_learn is None:
        agent_learn = make_dqn_agent(optim)

    if agent_opponent is None:
        agent_opponent = make_dqn_agent(optim)
        # agent_opponent = RandomPolicy()

    agents = [agent_opponent, agent_learn]
    policy = MultiAgentPolicyManager(agents, env)
    return policy, env.agents

# def make_ppo_agent() -> BasePolicy:
#     # model
#     net_a = Net(
#         args.state_shape,
#         hidden_sizes=hidden_sizes,
#         activation=torch.nn.Tanh,
#         device=device,
#     )
#     actor = ActorProb(
#         net_a,
#         action_shape,
#         unbounded=True,
#         device=device,
#     ).to(device)
#     net_c = Net(
#         state_shape,
#         hidden_sizes=hidden_sizes,
#         activation=torch.nn.Tanh,
#         device=device,
#     )
#     critic = Critic(net_c, device=device).to(device)
#     actor_critic = ActorCritic(actor, critic)

#     torch.nn.init.constant_(actor.sigma_param, -0.5)
#     for m in actor_critic.modules():
#         if isinstance(m, torch.nn.Linear):
#             # orthogonal initialization
#             torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
#             torch.nn.init.zeros_(m.bias)
#     # do last policy layer scaling, this will make initial actions have (close to)
#     # 0 mean and std, and will help boost performances,
#     # see https://arxiv.org/abs/2006.05990, Fig.24 for details
#     for m in actor.mu.modules():
#         if isinstance(m, torch.nn.Linear):
#             torch.nn.init.zeros_(m.bias)
#             m.weight.data.copy_(0.01 * m.weight.data)

#     optim = torch.optim.Adam(actor_critic.parameters(), lr=lr)

def make_dqn_agent(optim: Optional[torch.optim.Optimizer]) -> BasePolicy:
    env = _get_env()
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
        hidden_sizes=[256, 128, 128, 128, 128, 256],
        device="cuda" if torch.cuda.is_available() else "cpu",
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    # net = ResNet(17, 18, BasicBlock, env.action_space.shape or env.action_space.n)
    if optim is None:
        optim = torch.optim.Adam(net.parameters(), lr=1e-4)
    agent_learn = DQNPolicy(
        model=net,
        optim=optim,
        discount_factor=0.99,
        estimation_step=3,
        target_update_freq=320,
    )
    return agent_learn

def _get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    # return PettingZooEnv(go_v5.env(board_size=9, komi=7.5)) # 9x9 with no bonus to white player
    return PettingZooEnv(tictactoe_v3.env())

if __name__ == "__main__":
    N_TRAIN_ENVS        = 32
    N_TEST_ENVS         = 100
    REPLAY_BUFFER_LEN   = 50_000
    BATCH_SIZE          = 128
    MAX_EPOCH           = 500 # Has early stopping, can stop as soon as stop_fn returns true
    RUNNING_AVERAGE_LEN = 10
    CHECKPOINT_FREQUENCY= 10
    EXPERIMENT_NAME     = 'dqn_ttt'

    logger = TensorboardLogger(torch.utils.tensorboard.SummaryWriter('log/go/'+EXPERIMENT_NAME))

    # ======== Step 1: Environment setup =========
    train_envs = SubprocVectorEnv([_get_env for _ in range(N_TRAIN_ENVS)])
    test_envs = SubprocVectorEnv([_get_env for _ in range(N_TEST_ENVS)])

    # seed
    seed = 1337
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)

    # ======== Step 2: Agent setup =========
    policy, agents = _get_agents()

    # ======== Step 3: Collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(REPLAY_BUFFER_LEN, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=BATCH_SIZE * N_TRAIN_ENVS)# save model  # batch size * training_num

    # ======== Step 4: Callback functions setup =========
    def save_best_fn(policy):
        model_save_path = os.path.join("log", "rps", EXPERIMENT_NAME, "policy.pth")
        os.makedirs(os.path.join("log", "rps", EXPERIMENT_NAME), exist_ok=True)
        torch.save(policy.policies[agents[1]].state_dict(), model_save_path)

    # Only called if logger is defined
    def save_checkpoint_fn(epoch, env_step, gradient_step):
        ckpt_path = None
        if epoch % CHECKPOINT_FREQUENCY == 0:
            ckpt_path = os.path.join("log", "go", "ckpt", EXPERIMENT_NAME, "policy_checkpoint_0_{:03d}.pth".format(epoch))
            ckpt_path2 = os.path.join("log", "go", "ckpt", EXPERIMENT_NAME, "policy_checkpoint_1_{:03d}.pth".format(epoch))
            os.makedirs(os.path.join("log", "go", "ckpt", EXPERIMENT_NAME), exist_ok=True)
            
            torch.save(policy.policies[agents[1]].state_dict(), ckpt_path2)
            torch.save(policy.policies[agents[0]].state_dict(), ckpt_path)

        return ckpt_path

    # Early stopping function
    last_rewards = deque(maxlen=RUNNING_AVERAGE_LEN)
    def stop_fn(mean_rewards):
        last_rewards.append(mean_rewards)
        if len(last_rewards) < RUNNING_AVERAGE_LEN: return False
        avg = (sum(last_rewards)/len(last_rewards))
        return avg >= 0.95 # Won 95% of last RUNNING_AVERAGE_LEN games
        
    def train_fn(epoch, env_step):
        policy.policies[agents[1]].set_eps(0.1)

    def test_fn(epoch, env_step):
        policy.policies[agents[1]].set_eps(0.05)

    def reward_metric(rews):
        return rews[:, 1]

    print("Beginning training!")

    # ======== Step 5: Run the trainer =========
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=MAX_EPOCH,
        step_per_epoch=1000,
        step_per_collect=N_TRAIN_ENVS*5,
        episode_per_test=15,
        batch_size=BATCH_SIZE,
        train_fn=train_fn,
        test_fn=test_fn,
        # stop_fn=stop_fn,
        # save_best_fn=save_best_fn,
        save_checkpoint_fn= save_checkpoint_fn,
        update_per_step=0.1,
        test_in_train=False,
        reward_metric=reward_metric,
        logger=logger,
        # No need for prints, tensorboard updates automatically every thiry seconds
        show_progress=False,
        verbose=False, 
    )

    # return result, policy.policies[agents[1]]
    print(f"\n==========Result==========\n{result}")