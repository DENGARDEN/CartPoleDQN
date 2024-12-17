
import random
from collections import deque
from typing import List, Tuple, Dict
from multiprocessing import Process

import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import gym
import matplotlib.pyplot as plt

from lib import Config, DQNTrainer, plot_dqn_train_result

import statistics


def train_dqn(config, steps=100000):
    trainer = DQNTrainer(config)
    result = trainer.train(steps)

    return result


# TD(0)
if torch.cuda.is_available():
    config_1step = Config(run_name="Plain", env_id="CartPole-v1", device="cuda", n_steps=1, verbose=True,
                          optim_kwargs={'lr': 0.0001}, buffer_size=2048, batch_size=64, eps_start=1.0,
                          discount_rate=0.99, train_freq=1, learning_starts=1024)
else:
    config_1step = Config(run_name="Plain", env_id="CartPole-v1", device="cpu", n_steps=1, verbose=True,
                          optim_kwargs={'lr': 0.0001}, buffer_size=2048, batch_size=64, eps_start=1.0,
                          discount_rate=0.99, train_freq=1, learning_starts=1024)

train_result = train_dqn(config_1step)
print("mean reward from recent 10 episodes: ", statistics.mean(train_result[-10:-1]))
print("\n\n")
plot_dqn_train_result(train_result, label="1-Step DQN", alpha=0.9)
plt.axhline(y=500, color='grey', linestyle='-')  # 500 is the maximum score!
plt.xlabel("steps")
plt.ylabel("Episode Reward")

plt.legend()
plt.title("Training Comparison between Techniques for DQN")
plt.show()

# TD(n)

if torch.cuda.is_available():
    config_nstep = Config(run_name="Plain", env_id="CartPole-v1", device="cuda", n_steps=3, verbose=True,
                          optim_kwargs={'lr': 0.0001}, buffer_size=2048, batch_size=64, eps_start=1.0,
                          discount_rate=0.99, train_freq=1, learning_starts=1024)
else:
    config_nstep = Config(run_name="Plain", env_id="CartPole-v1", device="cpu", n_steps=3, verbose=True,
                          optim_kwargs={'lr': 0.0001}, buffer_size=2048, batch_size=64, eps_start=1.0,
                          discount_rate=0.99, train_freq=1, learning_starts=1024)

train_result = train_dqn(config_nstep)
print("mean reward from recent 10 episodes: ", statistics.mean(train_result[-10:-1]))
plot_dqn_train_result(train_result, label="3-Step DQN", alpha=0.9)
plt.axhline(y=500, color='grey', linestyle='-')  # 500 is the maximum score!
plt.xlabel("steps")
plt.ylabel("Episode Reward")

plt.legend()
plt.title("Training Comparison between Techniques for DQN")
plt.show()
