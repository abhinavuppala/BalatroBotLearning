---
layout: default
title: Status
---

[![Status Report Video](https://youtu.be/YFJ8Lg-jCeE)](https://youtu.be/YFJ8Lg-jCeE)

## Summary

Our project aims to create an agent using reinforcement learning to play and beat the hit game Balatro. Balatro is a single-player deck builder game centered around poker hands and involves strategy in building the optimal deck and finding the best synergizing set of “Joker” cards. This synergy allows players to achieve higher scoring combinations and progress further in the game.

The input to the model will be the game state, which covers two main phases of the game: playing hands during game rounds, and buying score-multiplying items during shop rounds.

The output of the agent will be a discrete decision from a large set of actions, in either of these two game states. This covers a wide variety of actions such as discarding some cards, playing a hand, buying a joker, or applying a tarot card (consumable) on certain cards in the deck.

Our project focuses on Single Agent Reinforcement Learning, using Proximal Policy Optimization (PPO). We currently use a botting script to allow our agent to play through the game and learn. We have also found a framework for simulating the game using a gymnasium-like environment, which we are currently working on configuring for our project; this should allow us to train much more efficiently.

## Approach

We use PPO with action masking, implemented with MaskablePPO from `Stable-Baselines3`. Masking is used because many of the actions are invalid during certain phases of the game; for instance, you cannot play cards during the shop phase, and likewise you cannot buy jokers during the blind phase.

At each update, we run our model for 128 steps, and collect a rollout of transitions. It uses its own current policy to play the game, update the policy based on rewards, and discards old data. Each rollout has the following values

$$
(s_t,a_t​,r_t​,V(s_t​),logπ(a_t​∣s_t​),mask_t​)
$$

Which represent the state, action, reward, next state (stored in the next element), value estimate, log probability of current action under current policy, and action mask.

The loss function we are optimizing is the usual PPO objective - the clipped surrogate objective. Clipping ensures that one single update doesn’t completely change the policy, which can help against overreacting to lucky actions.

After completing 128 steps, we compute advantages using Generalized Advantage Estimation (GAE), and returns for the value function. This is computed as

$$
\begin{aligned}
δ_t​=r_t​+γV(s_t+1​)−V(s_t​) \\
A^t=δ_t+γλδ_t+1+(γλ)2δ_t+2+…
\end{aligned}
$$

With a fairly standard γ=0.99 and λ=0.95.

Due to time constraints, as the training pipeline took quite a while to set up, we were only able to train for 9,600 trainsteps. At n_steps = 128 (128 steps per rollout), that means we were able to perform 75 policy updates.

Hyperparameters
* Learning rate = 3e-4 (default for SB3 PPO)
* N_steps = 128
* Batch size = 64
* N_epochs = 10
* Gamma = 0.99 (discount factor)
* GAE Lambda = 0.95 (standard for GAE)
* PPO clip_range = 0.2
* Policy = [128, 128] each (hidden layer sizes)

Reward Scaling Hyperparameters (configurable through command line args)
* Total timesteps = 9600
* Round clear reward = 1
* Chip Scaling reward = 0.001
* Game lost reward = -1
  
The input to our model is a JSON object of the full game state. We use a state encoder to encode it into a flat list of floats representing information like hand cards, current round scalars, shop features, money, etc.

We currently make a Gymnasium wrapper around the Balatro botting API, and use a single Discrete(274) space as our action space - the max number of actions possible in any context (56 play combos + 218 discard combos). Invalid actions are masked out with a boolean vector calculated using the game state from the custom function action_mask(). This allows us to restrict only valid actions; for instance, not buying items that are too expensive.

As mentioned above, we use a few hyperparameters for reward calculation. Whenever the agent scores points, it gets a small amount of proportional reward. Additionally, making it to the next round gives it a much bigger flat +1 reward, while losing the game results in a -1 reward.

This reward function helped the model improve slightly and more consistently reach round 2. However, this is still quite far from our target goal.

## Evaluation

For evaluation, we have some custom metrics such as mean chips scored per rollout and max round reached per rollout.

![Max Round Reached Graph](imgs\cs175_statusreport_chart1.png)

We use max round reached as a way to observe how close the agent is to beating the game, as fundamentally, the more rounds completed means the closer the agent is to winning the game. In our case here, we can see it starts off as mostly ending at Round 1 every time, and gradually it more consistently is able to reach Round 2.

Additionally, we use qualitative evaluation by watching the agent play the game, which can reveal trends harder to see otherwise. For instance, introducing a reward for playing chips makes the agent always play hands rather than discarding, in order to get an immediate reward. However, we need the model to learn some kind of delayed gratification, since high-value hands like a flush or full house are far more valuable than something like a high card, scoring around 300 vs. 15 points respectively.

![Mean Chips Graph](imgs\cs175_statusreport_chart2.png)

We use mean chips per rollout as a way to more accurately measure the agent’s performance in game. Since chips scale higher as rounds go up, the two metrics are related; however, mean chips provide a more specific look at the performance. For instance, here it is much more clearer that the model is improving and scoring more chips per round.

## Remaining Goals and Challenges

The biggest challenge we expect leading up to our final report is infrastructure.

Our current prototype is much more limited than the final goal. For starters, we began writing our botting script on a different API. When translated to the current API we decided to work with, it behaved differently, leading to more bugs and difficulty in gathering data. For our end result, we want to define its behavior clearly so that it handles every game state well. 

At the same time as using the botting API setup, we are looking into using a gymnasium-simulated environment for a training pipeline. However, this is proving to be quite difficult due to lack of documentation among other issues. Furthermore, getting it working on HCP3 would be another challenge following this. 

Another issue would be having enough time to produce meaningful results. Even with the faster training-loops provided by Gymnasium, the amount of time needed to create a well-performing agent is unpredictable, and we worry that there won’t be enough time to do this on top of creating visuals to evaluate our data.

If time permits, we also plan to look into other methods like Deep Q Learning. Since DQN is off-policy, it’s able to look to past experiences to influence the current policy, allowing us to add imagined experience and encourage a more effective playstyle.

## Resources Used

Botting API Resource: https://github.com/coder/balatrobot

RL Environment & Resources: https://github.com/giewev/balatrobot

Stable-Baselines3 Documentation: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html 

Schulman et al., “High-Dimensional Continuous Control Using Generalized Advantage Estimation,” 2016 (GAE).

Schulman et al., “Proximal Policy Optimization Algorithms,” 2017.

![Jimbo](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSzTR4gcn4KJXUa5gH4GNY_Txw0uQLZzDb1Aw&s)
