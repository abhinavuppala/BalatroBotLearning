---
layout: default
title: Proposal
---


[![Balatro Explained in 5 Minutes](https://www.youtube.com/watch?v=EFEauLTZZJ0)](https://www.youtube.com/watch?v=EFEauLTZZJ0)


## Summary

Our project aims to create an agent using reinforcement learning to play and beat the hit game **Balatro**. Balatro is a single-player deck builder game built around the idea of poker hands, where players must draw and play hands to beat increasing score thresholds. There is a lot of strategy involved in building the optimal deck and finding the best synergizing set of “Joker” cards which provide powerful modifiers, allowing for massive scaling.

The **input** to the model will be the game state, which Balatro has two main types of. First is during **rounds**, when the player plays poker hands to earn points & beat the current blind. The second is during **shops**, when the player buys modifiers and Joker cards to improve their scoring. Since the agent makes informed decisions based on a partially observable state, Balatro can be considered a **Partially observable Markov decision process (POMDP).**

The **output** of the agent will be a decision in either of these two phases, which covers a wide variety of actions such as discarding some cards, playing a hand, buying a joker, or applying a tarot card (consumable) on certain cards in the deck.

Our project focuses on **Cooperative Multi Agent Reinforcement Learning (MARL)** and our goal is to create a bot that can achieve high round scores and make it as far through the game as possible. We want our agent to be able to make intelligent decisions that reach a balance of short-term survivability and long-term scaling.

## Project Goals

- Minimum Goal: Agent **beats Ante 1 50%** of the time
- Realistic Goal: Agent **beats Ante 4 10%** of the time
- Moonshot Goal: Agent **beats Ante 8 10%** of the time

## AI/ML Algorithms

We plan to use **model-free reinforcement learning**, mainly using **Multi-Agent Proximal Policy Optimization (PPO)** but also trying other approaches such as **Single Agent Hierarchical RL**.

In order to effectively represent large game states (for instance, there exist 150 unique Joker cards), we will have to use methods like statistical abstraction and **vector embeddings** to compress the game state without losing information.

## Evaluation Plan

### Quantitative:

Some metrics we’re using to evaluate our implementation include the number of **rounds** beat, the **Ante** reached, the max/average **score** achieved, and the amount of **money** collected throughout the game. In terms of experimentation, we plan to run our MARL model and train it by using the above metrics to reward certain behaviors and choices.

### Qualiative:

Initial testing will include correct behavior of the model in the **environment** - playing cards correctly and accurately capturing the game environment. As the model trains, we will use libraries such as matplotlib to create **visualizations** of the model’s error as well as charts of the score’s progression approx. epochs. A successful one is a model that is able to **play** a turn, accept **feedback** based on what it plays, and make a productive **adjustment** to the game strategy. Graphs and charts will always visualize points where the model’s performance degrades or improves. 

## AI Tool Usage

- ChatGPT for brainstorming, debugging.

## Balatro Terms Reference

Ante: The level or stage in a game of Balatro from 1-8. 

Blind: A score goal that must be met in order to progress through an Ante.  

Joker Cards: Unique items that give players different abilities and score multipliers.  

Tarot Cards: Consumables that allow players to modify their cards and deck.  