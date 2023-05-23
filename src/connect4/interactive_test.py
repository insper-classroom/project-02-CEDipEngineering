from pettingzoo.classic import connect_four_v3
from agents import make_dqn_agent, make_ppo_agent
from PIL import Image, ImageTk
import tkinter as tk
import torch


SHAPE = 7
WINDOW_SIZE = (700,600)
DETERMINISTIC = False

class ImageWindow:
    def __init__(self, master, env):
        self.master = master
        self.canvas = tk.Canvas(master)
        self.canvas.pack(fill='both', expand=True)
        self.image = None
        self.photo_image = None
        self.canvas.bind('<Button-1>', self.on_click)
        self.game_on = True
        
    def load_image(self, image):
        # ROI zoom into actual board
        image = image[36:-35, 36:-35]

        # Red border on gameover
        if not self.game_on: 
            image[:20, :] = [255, 0, 0]
            image[:, :20] = [255, 0, 0]
            image[-20:, :] = [255, 0, 0]
            image[:, -20:] = [255, 0, 0]
        
        # Draw Image on canvas
        self.image = Image.fromarray(image)
        self.image = self.image.resize(WINDOW_SIZE, Image.Resampling.NEAREST)
        self.photo_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=self.photo_image, anchor='nw')
            
    def on_click(self, event):
        # If game is over, leave screen to view results
        if not self.game_on: return

        # Draw each step of game
        if self.image:
            action = (event.x-5)//100
            env.step(action)
            obs, r, term, trunc, _ = env.last()
            if r != 0:
                print("Game Over! You win!")
                # print("Rewards: {}".format(r))
                self.game_on = False
                self.load_image(env.render())
                return
            
            env.step(model_action_ppo(obs, DETERMINISTIC))
            obs, r, term, trunc, _ = env.last()
            if r != 0:
                if term or trunc:
                    print("You win! The AI made an illegal action!")
                else:
                    print("Game Over! You lose!")
                # print("Rewards: {}".format(r))
                self.game_on = False
                self.load_image(env.render())
                return
            
            self.load_image(env.render())

def model_action_dqn(obs: dict) -> int:
    global agent
    mask = torch.Tensor(obs["action_mask"])
    obs = torch.Tensor(obs["observation"]).unsqueeze(0)
    action = mask * agent.model(obs)[0]
    return torch.argmax(action).item()

def model_action_ppo(obs: dict, deterministic: bool = True) -> int:
    global agent
    mask = torch.Tensor(obs["action_mask"])
    obs = torch.Tensor(obs["observation"]).unsqueeze(0)
    logits, _ = agent.actor({"obs":obs})
    if not deterministic:
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()
    else: 
        action = logits.argmax(-1).item()
    print(logits, action)
    return action

if __name__ == '__main__':


    # Start env
    env = connect_four_v3.env(render_mode="rgb_array")
    env.reset()

    # log/c4/sppo_vs_pre/ckpt/21-05-23-22_49_49_ckpt_1.pth
    # log/c4/sppo_vs_pre/ckpt/21-05-23-22_49_49_ckpt_0.pth

    # Make agent    
    agent = make_ppo_agent(softmax=True)
    state_dict = torch.load("log/c4/sppo_vs_pre/ckpt/21-05-23-22_49_49_ckpt_0.pth")
    agent.load_state_dict(state_dict)

    # Agent starts
    # obs, *_ = env.last()
    # env.step(model_action_ppo(obs, deterministic=DETERMINISTIC))

    # Draw window
    root = tk.Tk()
    root.geometry('{}x{}'.format(*WINDOW_SIZE))
    root.title('Connect 4')
    
    # Make ImageWindow Viewermodel_action_ppo
    viewer = ImageWindow(root, env)

    # Draw env
    viewer.load_image(env.render())
    
    # Run tkinter app.
    root.mainloop()
