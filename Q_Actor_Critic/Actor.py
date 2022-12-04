import torch
import torch.nn as nn
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, layer_sizes, device):
        super(Actor, self).__init__()
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

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action).unsqueeze(0)
