from torch import Tensor

from . import *
from typing import Tuple
from collections import deque
import numpy as np

from collections import namedtuple
from random import sample
import torch

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done_mask"])


class ReplayMemory:
    def __init__(
            self,
            observation_shape: tuple = (),
            action_shape: tuple = (),
            buffer_size: int = 50000,
            num_steps: int = 1,
    ):
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.buffer_size = buffer_size
        self.num_steps = num_steps  # 구현해야 함

        self.memory = deque(maxlen=self.buffer_size)

    """TD(0)"""

    def write(self, state, action, reward, next_state, done_mask):
        # if self.num_steps > 1:

        transition = Transition(state=state, action=action, reward=reward, next_state=next_state, done_mask=done_mask)
        self.memory.append(transition)

    # 버퍼에서 Uniform Random으로 Transition들을 뽑습니다.
    def sample(self, num_samples: int = 1) -> Transition:
        states = []
        actions = []
        rewards = []
        next_states = []
        done_masks = []

        mini_batch = sample(self.memory, num_samples)

        for transition in mini_batch:
            state = transition.state
            action = transition.action
            reward = transition.reward
            next_state = transition.next_state
            done_mask = transition.done_mask
            # done = 1.0 if transition.done else done = 0.0

            states.append(state)
            actions.append([action])
            rewards.append([reward])
            next_states.append(next_state)
            done_masks.append([done_mask])

        return torch.tensor(states, dtype=torch.float), torch.tensor(actions), torch.tensor(rewards), torch.tensor(
            next_states, dtype=torch.float), torch.tensor(done_masks)

    def multi_step_sample(self, num_samples: int = 1, discount: float = 0.98, n_step: int = 2) -> Transition:
        states = []
        actions = []
        rewards = []
        next_states = []
        done_masks = []

        mini_batch = sample(list(enumerate(self.memory)), num_samples)

        for transition in mini_batch:
            idx, val = transition
            last_idx = idx + self.num_steps if idx + self.num_steps < self.memory.__len__() else idx

            state = val.state

            action = val.action

            # accumulate rewards to the last state
            reward = 0
            i = 0
            try:
                for step in range(n_step):
                    reward += self.memory[last_idx + step].reward * (discount ** step)
                    i = step
            except Exception as E:
                pass
            next_state = self.memory[last_idx].state
            done_mask = val.done_mask
            # done = 1.0 if transition.done else done = 0.0

            states.append(state)
            actions.append([action])
            rewards.append([reward])
            next_states.append(next_state)
            done_masks.append([done_mask])

        return torch.tensor(states, dtype=torch.float), torch.tensor(actions), torch.tensor(rewards), torch.tensor(
            next_states, dtype=torch.float), torch.tensor(done_masks)

    def __len__(self):
        return len(self.memory)

    """TD(n)"""
    # def write
