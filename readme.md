# Prompted Policy Search: Reinforcement Learning through Linguistic and Numerical Reasoning in LLMs


This repo serves as the code base for Prompted Policy Search (ProPS and ProPS<sup>+</sup>). The project website is [here](https://props-llm.github.io/).

<p align="center">
<img src = "static/banner.gif" width ="800" />
</p>

## Key Takeaways
In this paper, we demonstrate that:
1. LLMs can perform numerical optimization for Reinforcement Learning (RL) tasks.
2. LLMs can incorporate semantics signals, (e.g., goals, domain knowledge, ...), leading to more informed exploraton and sample-efficient learning.
3. Our proposed ProPS outperforms all baselines on 8 out of 15 Gymnasium tasks.


# Getting Started

## Install RL Tasks

- The RL tasks are based on gymnasium. Please install according to `https://github.com/Farama-Foundation/Gymnasium`
- There are 2 customized environments in the folders `./envs/gym-maze-master` and `./envs/gym-navigation-main`. If you want to train the maze or navigation agent, please pip install the packages.

## Install the LLM APIs

We utilized the standard Google Gemini, Openai, and Anthropic APIs. Please install the packages accordingly.

- `https://ai.google.dev/gemini-api/docs`
- `https://platform.openai.com/docs/overview`
- `https://docs.anthropic.com/en/release-notes/api`

## Start Training
In order to run an experiment, please run `python main.py --config <configuration_file>`.
