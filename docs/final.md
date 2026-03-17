---
layout: default
title: Final Report
---

<iframe width="560" height="315" src="https://www.youtube.com/embed/YFJ8Lg-jCeE?si=sUlCtCSi5zOUut5I" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

Couldn't figure out how to get it properly embedded

## Summary

Our project aims to create an agent using reinforcement learning to play and beat the hit game Balatro. Balatro is a single-player deck builder game centered around poker hands and involves strategy in building the optimal deck and finding the best synergizing set of “Joker” cards. This synergy allows players to achieve higher scoring combinations and progress further in the game.

The input to the model will be the game state, which covers two main phases of the game: playing hands during game rounds, and buying score-multiplying items during shop rounds. The game state has some aspect of randomness to it, such as what items are available to buy or if a specific effect will proc.

The output of the agent will be a discrete decision from a large set of actions, in either of these two game states. This covers a wide variety of actions such as discarding some cards, playing a hand, buying a joker, or applying a tarot card (consumable) on certain cards in the deck.

Our project focuses on Multi Agent Reinforcement Learning, using Proximal Policy Optimization (PPO). Initially we used a botting script to have the model read game state from the actual game for training; however, this proved to be very slow so we pivoted to a simulated gym environment method.

### Project Goals

- [✓] Minimum Goal: Agent **beats Ante 1 50%** of the time
- [✓] Realistic Goal: Agent **beats Ante 4 10%** of the time
- [✕] Moonshot Goal: Agent **beats Ante 8 10%** of the time

## Approach

We used two main approaches throughout our development. Initially, we used the BalatroBot API and ran the game locally for training, using the actual game state to train our model. While this worked, it was very slow, bounded by the game and our laptop's speed as it couldn't be run on HPC3.

Later, we pivoted to a simulation approach using a gymnasium API for Balatro simulation, which we were able to get working locally and on HPC3. This proved to be much more effective and got us much better results.

### Approach 1 - BalatroBot API

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

Reward Scaling Parameters (configurable through command line args)
* Total timesteps = 9600
* Round clear reward = 1
* Chip Scaling reward = 0.001
* Game lost reward = -1
  
The input to our model is a JSON object of the full game state. We use a state encoder to encode it into a flat list of floats representing information like hand cards, current round scalars, shop features, money, etc.

We currently make a Gymnasium wrapper around the Balatro botting API, and use a single Discrete(274) space as our action space - the max number of actions possible in any context (56 play combos + 218 discard combos). Invalid actions are masked out with a boolean vector calculated using the game state from the custom function action_mask(). This allows us to restrict only valid actions; for instance, not buying items that are too expensive.

As mentioned above, we use a few hyperparameters for reward calculation. Whenever the agent scores points, it gets a small amount of proportional reward. Additionally, making it to the next round gives it a much bigger flat +1 reward, while losing the game results in a -1 reward.

This reward function helped the model improve slightly and more consistently reach round 2. However, this is still quite far from our target goal.

### Approach 2 - Gymnasium Simulation

Similar to the botting version, we used PPO with action masking. However this time, we used a multi agent setup with Ray RLlib. This is because the two modes of the game we keep switching between - shop and blinds - are fundamentally different in their navigation and goals.

Also like the botting version, we have a similar training batch approach. The PPO equation is similar, and we also use clipping here, again to avoid one lucky run from completely overrunning the policy. This encourages the model to find more consistent means to win rather than just going between moonshot strategies.

We use GAE here similarly to the botting version after each train batch; however, since we have two different agents, we have some differences in hyperparameters here.

Overall we have $\gamma = 0.99$, and for:
* `shop_agent` $\lambda = 0.95$
* `blind_agent` $\lambda = 0.99$

Hyperparameters
* Learning rate = 1e-4
* Train batch size = 32768 = 2^15
* Minibatch size = 2048 = 2^11
* Num epochs = 3
  * Each train step, we do 3 passses over 2^15 samples from the train step, and each pass does 16 gradient updates (2^15 / 2^11 = 2^4). So 48 gradient steps per iteration
* Gamma & GAE Lambda above
* PPO clip_range = 0.3
  * Higher clip range than botting API (0.3 vs 0.2) means good/bad rounds sway the policy more. This allows faster learning

Reward Scaling Parameters
* Round won = 1.0
* Run win = 0.2
* Cash gained = 0.001
* Chips scored = 0.001 per chip scored

Action space
* Blind: `MultiBinary([6, hand_size+2])`
  * First we choose play vs. discard, then choosing which cards from the hand to play or discard
* Shop: Custom masked space depending on shop phase

We trained 3 different bots using these hyperparameters as starting points, with the following thought processes

* **Bot 1:** Balanced - the aim was to beat the early game and mainly just learn to use mechanics rather than doing anything too clever.
* **Bot 2:** Future-Thinking - this time, we focused more on future potential rewards and using the jokers more effectively.
  * `jokers_in_hand_attention = False -> True` so it would pay more attention to jokers it had when playing hands
  * `discard_potential_reward = 0.0 -> 0.1` to reward discarding when it results in it getting a better future hand
* **Bot 3:** Synergizing - this time, we aimed to encourage more long-term point scoring by rewarding joker synergies.
  * `joker_synergy_bonus = 0.0 -> 0.1` to reward jokers that play better together (ex: jokers that give additive & multiplicative mult)
  * `rarity_bonus = 0.0 -> 0.1` to reward rarer jokers as they tend to be better/more specialized

## Evaluation

### Approach 1 - BalatroBot API

For evaluation, we have some custom metrics such as mean chips scored per rollout and max round reached per rollout.

![Max Round Reached Graph](imgs\cs175_statusreport_chart1.png)

We use max round reached as a way to observe how close the agent is to beating the game, as fundamentally, the more rounds completed means the closer the agent is to winning the game. In our case here, we can see it starts off as mostly ending at Round 1 every time, and gradually it more consistently is able to reach Round 2.

Additionally, we use qualitative evaluation by watching the agent play the game, which can reveal trends harder to see otherwise. For instance, introducing a reward for playing chips makes the agent always play hands rather than discarding, in order to get an immediate reward. However, we need the model to learn some kind of delayed gratification, since high-value hands like a flush or full house are far more valuable than something like a high card, scoring around 300 vs. 15 points respectively.

![Mean Chips Graph](imgs\cs175_statusreport_chart2.png)

We use mean chips per hand as a way to more accurately measure the agent’s performance in game. Since chips scale higher as rounds go up, the two metrics are related; however, mean chips provide a more specific look at the performance. For instance, here it is much more clearer that the model is improving and scoring more chips per round.

### Approach 2 - Gymnasium Simulation

![Mean Round Graph](imgs\final_round_mean.png)

As seen from the mean round graph for the gymnasium simulation vs. the balatrobot API version, we achieved a far better result on either of the 3 bots here than compared to the botting one. The improved training speed & ability to use HPC3 with the simulated approach exponentially improved our training time, allowing us to do over 1k total trainsteps per bot.

The graph shows bot 1, 2, and 3 eventually all reaching an average round of about 6 (ie. beating Ante 2 on average). This means we beat our baseline goal of beating Ante 1 50% of the time! As we can see, bot 2 and 3 reached the average round 6 much faster but stagnated around there, while bot 1 gradually reached there.

![Mean Chips Graph](imgs\final_chips_mean.png)

However, the mean chips scored shows a different story. While bot 2 and 3 are stuck around 3000 chips on average, bot 1 consistently increased, averaging around 8-9000 chips scored with a lot of fluctuation.

Since chips scale exponentially with respect to round in Balatro, we can infer what happened with bot 1 vs bots 2 & 3. Since bot 1 has a much higher chips scored average than bot 2, it likely means that bot 1 has more rounds that went much further than any games with bot 2 or 3 did, while it also had a lot more rounds that made it much less far.

Essentially, bot 2 and 3 are much more consistent in the early game but it's much rarer that it makes it past then. However, while bot 1 loses early quite often, when it's able to get a good run going, it makes it much further.

![Round 12 (Ante 4) Won Mean](imgs\round12_won_mean.png)

![Round 24 (Ante 8) Won Mean](imgs\round24_won_mean.png)

These graphs support the claim above, as bot 1 beat ante 4 around 20% of the time while bot 2 and 3 only beat ante 4 around 2-3% of the time. Similarly, bot 1 is able to beat ante 8 (ie. beat the game) around 2% of the time, while bot 2 and 3 only do this maybe 0.4% of the time.

## Insights & Future Improvements

Our realistic goal was to beat Ante 4 (round 12) 10% of the time, which we achieved with bot 1 as we got around a 20% winrate of Ante 4.

However, we weren't able to achieve our moonshot goal of beating Ante 8 10% of the time; we only got to around a 2% winrate here.

The biggest challenges we faced throughout the project was environment setup challenges, as we had to try many different versions and training pipelines to finally get something working. As a result, we didn't have as much time as we would've liked to fine-tune hyperparameters or reward functions.

One such improvement could be an adaptive reward function that puts more weight to general joker purchases early game, while putting more weight to joker synergies late-game. The biggest weakness of bot 2 and 3 is that they weren't able to get good runs off the ground and running because they were too picky with jokers or hands played. As a result they had less opportunities to look for game-winning jokers, while bot 1 would buy whatever it could, getting it past early game and giving it more opportunities to find better jokers. If we could combine the best of both worlds, we might be able to achieve a better winrate.

If time permits, we could also look into other methods like Deep Q Learning. Since DQN is off-policy, it’s able to look to past experiences to influence the current policy, allowing us to add imagined experience and encourage a more effective playstyle from real good Balatro players.

## Resources Used

Botting API Resource: https://github.com/coder/balatrobot

RL Environment & Resources: https://github.com/giewev/balatrobot

Stable-Baselines3 Documentation: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html 

Schulman et al., “High-Dimensional Continuous Control Using Generalized Advantage Estimation,” 2016 (GAE).

Schulman et al., “Proximal Policy Optimization Algorithms,” 2017.

![Jimbo](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSzTR4gcn4KJXUa5gH4GNY_Txw0uQLZzDb1Aw&s)
