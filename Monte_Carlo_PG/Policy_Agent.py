import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, layer_sizes):
        super(Policy, self).__init__()
        self.layers = self.build_nn(layer_sizes)

    def build_nn(self, layer_sizes):
        assert len(layer_sizes) > 1
        layers = []
        for index in range(len(layer_sizes)-1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index+1])
            act =    nn.ReLU() if index < len(layer_sizes)-2 else nn.Softmax()
            layers += (linear,act)
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
