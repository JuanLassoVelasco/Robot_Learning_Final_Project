import torch
import torch.nn as nn
from torch.distributions import Categorical

class Critic(nn.Module):
    def __init__(self, layer_sizes, device):
        super(Critic, self).__init__()
        self.device = device
        self.layers = self.build_nn(layer_sizes=layer_sizes)

    def build_nn(self, layer_sizes):
        assert len(layer_sizes) > 1
        layers = []
        for index in range(len(layer_sizes)-1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index+1])
            act =    nn.ReLU() if index < len(layer_sizes)-2 else nn.Softmax()
            layers += (linear,act)
        return nn.Sequential(*layers)

    def forward(self, state):
        return self.layers(state)

    def critique(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.forward(state).cpu()
        # m = Categorical(probs)
        # action = m.sample()
        return probs
