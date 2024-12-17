import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import *

import random


class SimpleMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[128, ]):
        super().__init__()

        # simple NN
        self.inp_layer = nn.Linear(input_size, hidden_sizes[0])
        self.fc1 = nn.Linear(hidden_sizes[0], hidden_sizes[0] * 2)
        self.fc2 = nn.Linear(hidden_sizes[0] * 2, hidden_sizes[0] * 2)
        # self.fc3 = nn.Linear(hidden_sizes[0] * 4, hidden_sizes[0] * 4)
        # self.fc4 = nn.Linear(hidden_sizes[0] **7, hidden_sizes[0] **7)
        self.out_layer = nn.Linear(hidden_sizes[0] * 2, output_size)

        # self.linears = nn.ModuleList(
        #     [
        #         nn.Linear(in_shape, out_shape) for in_shape, out_shape in zip(hidden_sizes[:-1], hidden_sizes[1:])
        #     ]
        # )
        # self.out_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x = F.relu(self.inp_layer(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))

        # for layer in self.linears:
        #     x = F.relu(layer(x))

        return self.out_layer(x)

    def e_greedy(self, observation, epsilon):
        pass
