# Reinforcement Learning in Simple Simulated Navigation Task
Hello, world!

This exploratory project includes the code of different reinforcement learning (RL) implementations to explore [the frozen lake example (FrozenLake-v0) by OpenAI Gym](https://gym.openai.com/envs/FrozenLake-v0/). 

# How to Run the Code
## Package Prerequisites
- ```gym==0.18.0``` ([OpenAI Gym](https://gym.openai.com/docs/#installation))
- ```numpy```

## Demo Code
Jupyter Notebook To Be Added!
## Command Line Usage

To explore more options, set the working directory to project root and print the help info by running

```
$ cd <path to project root>
$ ./src/MC.py -h
```

### Default: Show Success Rate Only

To set the working directory to project root and run the default settings of Monte Carlo simulation and see the final rate of succuess (accuracy) of each of the policy trained in each round, use command

```
$ cd <path to project root>
$ ./src/MC.py
```

This will train one policy for each round, displaying policy success rate at the end of the round, for a total of 4 rounds. 

### Visualization: Show Interactions

There are other training settings available. The following command displays agent-environment interaction every 1000 (instead of the default 2000) training episodes, for a total of 4000 episodes (instead of the default 5000) in each round, for 1 round (instead of the default 4):

```
$ cd <path to project root>
$ ./src/MC.py --display --display_every 1000 --episodes 4000 --rounds 1
```

You will see maps being printed out every 3000 epochs trained. They will resemble the following:

```
SFFFFFFF
FFFFFFFF
FFFHFFFF
FFFFFHFF
FFFHFFFF
FHHFFFHF
FHFFHFHF
FFFHFFFG
  (Right)
```

where S = start (safe), F = frozen (safe), H = hole (lose), G = goal (win). The red cursor highlighting one of the four characters on the screen shows the agent's current location resulted from the agent's last action, shown in the bottom right corner.

# RL Algorithms Explored

## Monte Carlo Estimation

Specifically, this project implements the On-Policy First-Visit Monte-Carlo Control Algorithm for ϵ-soft Policies. For pseudocode, see Figure 5.6 on p.127 on Sutton & Barto.

## SARSA Temporal Difference (TD) Learning

To Be Added!

## Q-Learning

To Be Added!

# Code Structure

Tentative structure, possible modification soon.
## Agent Class
Each agent has its own policy, and can act (forward) and update (backward). 
## Environment Class

TBD!

# Acknowledgements & Resources
I have put together inspirations from multiple sources, without which this project cannot actualize. 

Besides all the StackExchange discussions, the algorithms and implementation ideas in this project are drawn especially from the following sources: 
* **a university course**: [Discourse and Dialogue](http://www.cs233.org), where I was introduced to RL application in dialogue generation
* **a tutorial**: [Introduction to RL in Python](https://www.coursera.org/projects/introduction-to-reinforcement-learning-in-python) on Coursera Project Network
* **a textbook**: [Introduction to RL](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) by Sutton & Barto (2nd ed., 2014-15 in progress); page numbers on this page corresponds to pages in this pdf file
* **some skeleton code**: [Source Code for RandomAgent Class](https://github.com/openai/gym/blob/master/examples/agents/random_agent.py)

# Style Notes
For learning purposes, the code in this repository will be more extensively commented than usual. 

