**GoAgent**
======================

This project's goal is to implement an agent to play the chinese boardgame [Go](https://en.wikipedia.org/wiki/Go), as implemented in the [Farama Petting Zoo environment](https://pettingzoo.farama.org/environments/classic/go/). Part of the goal of this project is to train this agent against another similar agent, which is learning alongiside it.

**Environment**
=====================

As can be seen in the Farama Petting Zoo documentation page, this environment is a wrapper for [MiniGo](https://github.com/tensorflow/minigo). As can be seen below, the game board consists of an orthogonal lattice of square tiles, upon whose line intersections pieces (called stones) are placed.

!['Gif of board game go'](/img/classic_go.gif)

The goal of the game is to capture stones of your oponent's color, which can be accomplished by completely surrounding them with your stones. The game can be played in many different board sizes, and rules differ greatly according to period and region, however most traditionally the game is played in 19x19 format, however beginners often play in a 9x9 or 13x13 configuration for a quicker match.

**Implementation**
======================

The implementation will utilize PyTorch to execute the training, and will possibly be made using the DDQN or Reinforcement policy models. The expected result is an agent capable of playing Go against an beginner/intermediate human player.


