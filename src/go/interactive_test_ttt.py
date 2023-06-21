from pettingzoo.classic import go_v5, tictactoe_v3
import matplotlib.pyplot as plt
import os
from typing import Optional, Tuple
from collections import deque

from resnet18 import ResNet, BasicBlock

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageTk
from tianshou_marl import make_dqn_agent
from tianshou.utils.net.common import Net


def model_action(obs: dict) -> int:
    global agent
    mask = torch.Tensor(obs["action_mask"])
    obs = torch.Tensor(obs["observation"]).unsqueeze(0)
    action = agent.model(obs)[0]
    return torch.argmax(action).item()


if __name__ == '__main__':


    # Start env
    env = tictactoe_v3.env(render_mode="human")
    env.reset()
    
    agent = make_dqn_agent(None)
    state_dict = torch.load("log/ckpt/dqn_ttt/policy_checkpoint_1_500.pth")
    agent.load_state_dict(state_dict)

    done = False

    while not done:


        obs, r, term, trunc, _ = env.last()
        action = int(input("Choose your action: "))
        env.step(action)

        r = env.rewards
        if sum(r.values()) != 0:
            print("Game Over!")
            print("Rewards: {}".format(r))
            break
    
        obs, r, term, trunc, _ = env.last()
        action = model_action(obs)
        env.step(action)

        r = env.rewards
        if sum(r.values()) != 0:
            print("Game Over!")
            print("Rewards: {}".format(r))
            break
