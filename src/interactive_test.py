from pettingzoo.classic import go_v5, tictactoe_v3
import matplotlib.pyplot as plt
import os
from typing import Optional, Tuple
from collections import deque

from resnet18 import ResNet, BasicBlock

import gymnasium as gym
import numpy as np
import torch
import tkinter as tk
from PIL import Image, ImageTk
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy

SHAPE = 9

class ImageWindow:
    def __init__(self, master, env):
        self.master = master
        self.canvas = tk.Canvas(master)
        self.canvas.pack(fill='both', expand=True)
        self.image = None
        self.photo_image = None
        self.canvas.bind('<Button-1>', self.on_click)
        
    def load_image(self, image):
        self.image = Image.fromarray(image)
        self.image = self.image.resize((800, 800), Image.Resampling.NEAREST)
        self.photo_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=self.photo_image, anchor='nw')
            
    def on_click(self, event):
        if self.image:
            x, y = self.convert_pixel_to_cell(event.x-10, event.y-10)
            action = self.convert_cell_to_env_action(x, y)
            # print("Clicked coordiantes: {}".format((event.x, event.y)))
            # print("Cell: {}".format(x, y))
            # print("Equivalent Action: {}".format(action))
            env.step(action)
            r = env.rewards
            if sum(r.values()) != 0:
                print("Game Over!")
                print("Rewards: {}".format(r))
                self.master.destroy()
                return
            
            obs, *_ = env.last()
            env.step(model_action(obs))

            r = env.rewards
            if sum(r.values()) != 0:
                print("Game Over!")
                print("Rewards: {}".format(r))
                self.master.destroy()
                return
            
            self.load_image(env.render())

    def convert_pixel_to_cell(self, x: int, y: int):
        return x // (800//SHAPE), y // (800//SHAPE)
    
    def convert_cell_to_env_action(self, x: int, y: int):
        return SHAPE*x + y

def model_action(obs: dict) -> int:
    global agent
    mask = torch.Tensor(obs["action_mask"])
    obs = torch.Tensor(obs["observation"]).unsqueeze(0)
    action = mask * agent.model(obs)[0]
    return torch.argmax(action).item()
    
def make_dqn_agent(optim: Optional[torch.optim.Optimizer]) -> BasePolicy:
    env = PettingZooEnv(go_v5.env(board_size=9, komi=7.5))
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gym.spaces.Dict)
        else env.observation_space
    )

    net = ResNet(17, 18, BasicBlock, env.action_space.shape or env.action_space.n)
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


if __name__ == '__main__':


    # Start env
    env = go_v5.env(board_size = 9, render_mode="rgb_array")
    # env = tictactoe_v3.env(render_mode="rgb_array")
    env.reset()
    
    agent = make_dqn_agent(None)
    state_dict = torch.load("log/ckpt/cdqn/policy_checkpoint_3000.pth")
    agent.load_state_dict(state_dict)

    obs, *_ = env.last()
    env.step(model_action(obs))

    # Draw window
    root = tk.Tk()
    root.geometry('800x800')
    root.title('Image Viewer')
    
    # Make ImageWindow Viewer
    viewer = ImageWindow(root, env)

    # Draw env
    viewer.load_image(env.render())
    
    # Run tkinter app.
    root.mainloop()
