# Agents

There are three off-policy RL algorithms included in this repo: 
Deep Q-Learning (DQN),
Deep Deterministic Policy Gradient (DDPG) and
Soft Actor Critic (SAC).
Proximal Policy Optimization (PPO) augmented by Generalized Advantage Estimation (GAE) is 
an on-policy RL algorithm included herein.

Off-policy RL algorithms have a separate memory which saves
all transitions collected under arbitrary policies in the
a memory or buffer. We provide a basic first-in-first-out 
buffer and a more advanced prioritized experience replay buffer.
For these types of algorithms, Hindsight Experience Replay (HER)
is implemented if the parameter "her_ratio" is greater 0.


## DQN
https://arxiv.org/pdf/1312.5602.pdf

## DDPG
https://arxiv.org/pdf/1509.02971.pdf

## SAC
https://arxiv.org/pdf/1801.01290.pdf

## HER
https://arxiv.org/pdf/1707.01495.pdf

## PPO
https://arxiv.org/pdf/1707.06347.pdf

