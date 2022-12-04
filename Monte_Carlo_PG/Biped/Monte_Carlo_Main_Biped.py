import sys
import time
import numpy as np
from collections import deque

import torch
import gym

sys.path.insert(1, '/Users/juanlasso/Documents/Robot_Learning_Final_Project/Monte_Carlo_PG/')

from Policy_Agent import Policy

from time import sleep
from gym.wrappers import record_video

ss_horizon = 10
ss_score_threshold = -100

env_id = 'MountainCar-v0'
env = gym.make(env_id)

s_size = env.observation_space.shape[0]
a_size = env.action_space.n

hid_layer_size = 64

learnRate = 1e-3
num_episodes = 3000
episode_timout = 1000
discount_rate = 0.99
score_disp_interval = 100

policy = Policy(layer_sizes=[s_size, hid_layer_size, a_size])

optimizer = torch.optim.Adam(policy.parameters(), lr=learnRate)

scores_deque = deque(maxlen=100)
scores = []

start_time = time.time()
ss_return_time = 0

horizon_ticker = 0

for episode in range(1, num_episodes+1):
    saved_log_probs = []
    rewards = []
    obs = env.reset()
    obs, _ = obs
    for t in range(episode_timout):
        action, log_prob = policy.act(obs)
        saved_log_probs.append(log_prob)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        rewards.append(reward)
        if done:
            break 

    tot_score = sum(rewards)
    scores_deque.append(tot_score)
    scores.append(tot_score)
    
    discounts = [discount_rate**i for i in range(len(rewards)+1)]
    R = sum([a*b for a,b in zip(discounts, rewards)])
    
    # Line 7:
    policy_loss = []
    for log_prob in saved_log_probs:
        policy_loss.append(-log_prob * R)
    policy_loss = torch.cat(policy_loss).sum()
    
    # Line 8:
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    
    if episode % score_disp_interval == 0:
        print('Episode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))

    if tot_score >= ss_score_threshold:
        horizon_ticker += 1
    else:
        horizon_ticker = 0

    if horizon_ticker > ss_horizon:
        ss_return_time = time.time()

end_time = time.time()

env_to_wrap = gym.make('MountainCar-v0', render_mode="rgb_array")
env = record_video.RecordVideo(env_to_wrap, 'video')
for i in range(2):
    obs, done, rew = env.reset(), False, 0
    obs, _ = obs
    while (done != True) :
        action, _ =  policy.act(obs)
        obs, reward, terminate, truncate, info = env.step(action)
        done = terminate or truncate
        rew += reward
        sleep(0.01)
    print("episode : {}, reward : {}".format(i,rew)) 

    print(rew)
    env.close()
    env_to_wrap.close()

final_train_time = end_time - start_time
threshld_reach_time = ss_return_time - start_time

print("Train time: " + str(final_train_time) + " Time to reach " + str(ss_score_threshold) + " score: " + str(threshld_reach_time))
