from time import sleep
import torch
import torch.nn as nn
import copy
import random
from collections import deque 

class DQN_Agent:
    
    def __init__(self, seed, layer_sizes, lr, sync_freq, exp_replay_size, action_space):
        torch.manual_seed(seed)
        self.action_space = action_space
        self.deepQ_net = self.build_nn(layer_sizes)
        self.target_net = copy.deepcopy(self.deepQ_net)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.deepQ_net.parameters(), lr=lr)
        
        self.network_sync_freq = sync_freq
        self.network_sync_counter = 0
        self.discnt_rate = 0.95
        self.stored_experience = deque(maxlen = exp_replay_size)  
        
    def build_nn(self, layer_sizes):
        assert len(layer_sizes) > 1
        layers = []
        for index in range(len(layer_sizes)-1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index+1])
            act =    nn.ReLU() if index < len(layer_sizes)-2 else nn.Identity()
            layers += (linear,act)
        return nn.Sequential(*layers)
    
    def get_action(self, state, action_space_len, epsilon):
        with torch.no_grad():
            Q_preds = self.deepQ_net(torch.from_numpy(state).float())
        action = torch.max(Q_preds, axis=0)
        action = action if torch.rand(1,).item() > epsilon else self.action_space.sample()
        return action
    
    def get_q_next(self, state):
        with torch.no_grad():
            Q_preds = self.target_net(state)
        Q_next,_ = torch.max(Q_preds, axis=0)    
        return Q_next
    
    def collect_experience(self, experience):
        self.stored_experience.append(experience)
        return
    
    def sample_from_experience(self, sample_size):
        if(len(self.stored_experience) < sample_size):
            sample_size = len(self.stored_experience)   
        sample = random.sample(self.stored_experience, sample_size)
        states = torch.tensor([exp[0] for exp in sample]).float()
        actions = torch.tensor([exp[1] for exp in sample]).float()
        rewards = torch.tensor([exp[2] for exp in sample]).float()
        next_state = torch.tensor([exp[3] for exp in sample]).float()  
        done =  torch.tensor([exp[4] for exp in sample]).float()
        return states, actions, rewards, next_state, done
    
    def train(self, batch_size ):
        state, action, reward, next_state, done = self.sample_from_experience(sample_size = batch_size)
        if(self.network_sync_counter == self.network_sync_freq):
            self.target_net.load_state_dict(self.deepQ_net.state_dict())
            self.network_sync_counter = 0
        
        # predict expected return of current state using main network
        Q_pred = self.deepQ_net(state)
        
        # get target return using target network
        Q_next = self.target_net(next_state)
        target_reward = reward + self.discnt_rate * Q_next
        
        loss = self.criterion(Q_pred, target_reward)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        
        self.network_sync_counter += 1       
        return loss.item()
