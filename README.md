[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/7Wj0oCgF)
**Connect Four Agent**
======================

This project's goal is to implement an agent to play the boardgame [Connect Four](https://en.wikipedia.org/wiki/Connect_Four), as implemented in the [Farama Petting Zoo environment](https://pettingzoo.farama.org/environments/classic/connect_four/). Part of the goal of this project is to train this agent against another similar agent, which is learning alongside it.

**Environment**
=====================

The connect four game consists of a grid of 6 rows and 7 columns, where pieces are dropped in an alternating fashion by each of the two players. The goal is to connect 4 of your own pieces, while preventing your opponent from doing the same. 

!['Gif of board game connect four'](/img/classic_c4.gif)

The available actions are numbered $0:6$, and refer to one column from left to right. When a piece is placed on a column, it falls to the lowest empty row in that column. Actions that select a column with 6 pieces are deemed illegal. 

Observations are supplied in the form of a $6x7x2$ matrix, wherein each position represents a slot on the board, and each plane refers to whether an agent has a piece on that slot.

The environment offers rewards only in the terminal steps, where the winning player gets +1 and the losing player gets -1. If a player performs an illegal action, they receive a reward of -1 (the adversary receives 0) and the episode terminates.

**Implementation**
======================

A library called Tianshou was found to implement a lot of the necessary functionality. Their docs are available in a [176 page pdf](https://tianshou.readthedocs.io/_/downloads/en/master/pdf/) and also in their [github](https://github.com/thu-ml/tianshou/tree/master).

The library offers many useful wrappers, such as an Subprocess Environment Vector wrapper, which automatically runs several environments in parallel (using python's multiprocessing library), and allows for batch prediction/training of the agent.

Another very useful utility was the training loop abstractions, allowing for both on-policy and off-policy training of models with very little alteration. It was also very helpful to have a built-in TensorBoard logger module, documenting every iteration of model training history.

In the end, two different agents were attempted, each trained against an equal and a random agent. The first was a Double Deep Q-Learning (DQN) agent, and the second was a Proximal Policy Optimization (PPO) agent.

**Results:**
======================

The first iteration of the project sought to train a DQN agent against another identical autonomous DQN agent. Below we can observe the training process.

!['DQN results'](/img/c4_dqn.png)

On the left, it is possible to see that the rewards converged somewhat towards 1 (victory for one agent), and on the right, the length of the episode seemed to increase through training, suggesting that the agents were getting better at stopping their adversary from winning.

After this, another agent was trained with the same network and parameters, however this time against a random agent.

!['DQN_rand results'](/img/c4_dqn_rand.png)

In this scenario, learning took place more slowly, and episode length did not seem to converge at all.

The second iteration of the project was done using PPO, and initially (like DQN), was made to pit two identical PPO agents against one another.

!['PPO results'](/img/c4_ppo.png)

Now the rewards tended heavily towards -1 (meaning one agent was winning every time), and episode length dropped to 7. Upon further examination, it was discovered that both players try to speed through the game placing every piece in the same column, until the player that started won (7 moves, starting player wins at their 4th move).

Then, to match DQN, PPO was trained against a random agent.

!['PPO_rand results'](/img/c4_ppo_rand.png)

In this case, PPO seemed to win pretty often, however it was still behaving rather poorly, possibly due to having trouble learning not to play illegal moves.

During these trainings, however, numerical errors started emerging, since the used PPO model was not using softmax. A new model was trained against a random player, this time using softmax at the end of the neural network.

!['SPPO_rand results'](/img/sppo_rand.png)

The player eventually learned to beat the random agent quite often, mostly by just stacking pieces vertically again.

Then this Softmax PPO agent's weights were copied onto a new agent, that was set to train against a fresh but equal PPO agent. The idea was to see if this new player could learn to defeat their sibling's repetitive strategy.

!['SPPO_pre results'](/img/sppo_pre.png)

This model struggled a bit at first, but eventually both started playing longer matches. Below you can see the resulting agent in action (probabilistic decisions are used during test as well).

!['gif model test'](/img/c4_example.gif)

**Closing Thoughts**
===========================

Training a model in a competitive environment is much harder than a standard single agent setting. That said, I believe I was able to achieve some success in making an agent that seems to have a grasp of the rules of the game. It is quite silly, and makes many mistakes, however if you allow it to play freely, it will always win.

**How to run:**
===========================

To run the training of an agent, simply run:

    $ python src/connect4/connect4.py

This file can be edited to change which agent is being trained, where the files are logged, and many other things described in the file itself.

To run an interactive test as seen in the gif above, run:

    $ python src/connect4/interactive_test.py

Note that this requires you to have trained a model first, and to edit the *MODEL_PATH* variable in the mentioned script.





