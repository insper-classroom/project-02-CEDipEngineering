"""
This is a minimal example of using Tianshou with MARL to train agents.

Author: Will (https://github.com/WillDudley)

Python version used: 3.8.10

Modified by: CEDip (https://github.com/CEDipEngineering)
"""

import os
from datetime import datetime
from collections import deque
from typing import Optional, Tuple, Callable


import numpy as np
import torch

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy, PPOPolicy
from tianshou.trainer import offpolicy_trainer, onpolicy_trainer
from tianshou.utils import TensorboardLogger

from agents import make_dqn_agent, make_ppo_agent
# from resnet18 import ResNet, BasicBlock

from pettingzoo.classic import connect_four_v3


def _get_multi_agents(
        agent_learn:    Optional[Callable[[None], BasePolicy]],
        agent_opponent: Optional[Callable[[None], BasePolicy]],
    ) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    
    env = _get_env()
    agents = [agent_opponent(), agent_learn()]
    policy = MultiAgentPolicyManager(agents, env)
    return policy, env.agents

_get_env = lambda : PettingZooEnv(connect_four_v3.env()) # This function is needed to provide callables for DummyVectorEnv.

if __name__ == "__main__":
    N_TRAIN_ENVS        = 40
    N_TEST_ENVS         = 100
    REPLAY_BUFFER_LEN   = 50_000    
    BATCH_SIZE          = 128
    MAX_EPOCH           = 50_000
    RUNNING_AVERAGE_LEN = 10
    CHECKPOINT_FREQUENCY= 10
    LOG_DIR             = 'log/c4/'
    EXPERIMENT_NAME     = 'sppo_vs_pre2'


    # ======== Step 1: Environment setup =========
    train_envs = SubprocVectorEnv([_get_env for _ in range(N_TRAIN_ENVS)])
    test_envs = SubprocVectorEnv([_get_env for _ in range(N_TEST_ENVS)])

    # Logger
    logger = TensorboardLogger(torch.utils.tensorboard.SummaryWriter(LOG_DIR+EXPERIMENT_NAME))
    
    # seed
    seed = 1337
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)


    # log/c4/sppo_vs_pre/ckpt/21-05-23-22_49_49_ckpt_1.pth
    # log/c4/sppo_vs_pre/ckpt/21-05-23-22_49_49_ckpt_0.pth

    # path1 = "log/c4/sppo_vs_rand/ckpt/21-05-23-01_20_52_ckpt_1.pth"
    # path2 = "log/c4/sppo_vs_rand/ckpt/21-05-23-01_20_52_ckpt_1.pth"


    path1 = "log/c4/sppo_vs_pre/ckpt/21-05-23-22_49_49_ckpt_1.pth"
    path2 = "log/c4/sppo_vs_pre/ckpt/21-05-23-22_49_49_ckpt_1.pth"

    sppo = make_ppo_agent(softmax=True)
    sppo.load_state_dict(torch.load(path1))

    sppo2 = make_ppo_agent(softmax=True)
    sppo2.load_state_dict(torch.load(path2))

    # ======== Step 2: Agent setup =========
    policy, agents = _get_multi_agents(
        agent_learn = lambda : sppo, 
        agent_opponent = lambda: sppo2, 
    )

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
        model_save_path = os.path.join(LOG_DIR, EXPERIMENT_NAME, "policy.pth")
        os.makedirs(os.path.join(LOG_DIR, EXPERIMENT_NAME), exist_ok=True)
        torch.save(policy.policies[agents[1]].state_dict(), model_save_path)

    # Only called if logger is defined
    def save_checkpoint_fn(epoch, env_step, gradient_step):
        ckpt_path = None
        if epoch % CHECKPOINT_FREQUENCY == 0:
            os.makedirs(os.path.join(LOG_DIR , EXPERIMENT_NAME, "ckpt"), exist_ok=True)
            
            fn1 = datetime.now().strftime(r'%d-%m-%y-%H_%M_%S_ckpt_1.pth')
            ckpt_path2 = os.path.join(LOG_DIR, EXPERIMENT_NAME, "ckpt", fn1)
            torch.save(policy.policies[agents[1]].state_dict(), ckpt_path2)
            
            fn0 = datetime.now().strftime(r'%d-%m-%y-%H_%M_%S_ckpt_0.pth')
            ckpt_path = os.path.join(LOG_DIR , EXPERIMENT_NAME, "ckpt", fn0)
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
    result = onpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=MAX_EPOCH,
        step_per_epoch=1000,
        step_per_collect=N_TRAIN_ENVS*5,
        episode_per_test=15,
        repeat_per_collect=4, # Number of times the same batch is fed through the learning pipeline
        batch_size=BATCH_SIZE,
        # train_fn=train_fn,
        # test_fn=test_fn,
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