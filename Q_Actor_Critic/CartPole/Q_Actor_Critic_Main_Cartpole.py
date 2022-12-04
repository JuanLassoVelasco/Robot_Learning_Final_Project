import gym
import sys
import time
import numpy as np
import torch
import torch.optim as optim
from collections import deque

sys.path.insert(1, '/Users/juanlasso/Documents/Robot_Learning_Final_Project/Q_Actor_Critic/')

from Actor import Actor
from Critic import Critic

from time import sleep
from gym.wrappers import record_video

ss_horizon = 10
ss_score_threshold = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v1").unwrapped

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
hid_layer_size = 64
lr = 0.0001
discnt_rt = 0.99

score_disp_interval = 100

def compute_returns(next_value, rewards, masks, gamma=discnt_rt):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns
    


actor = Actor([state_size, hid_layer_size, action_size], device=device)
critic = Critic([state_size, hid_layer_size, action_size], device=device)
num_episodes = 3000
episode_timout = 1000

scores_deque = deque(maxlen=100)

optimizerA = optim.Adam(actor.parameters())
optimizerC = optim.Adam(critic.parameters())

start_time = time.time()
ss_return_time = 0

horizon_ticker = 0

for iter in range(num_episodes):
    done = False
    log_probs = []
    values = []
    rewards = []
    masks = []
    state = env.reset()
    state, _ = state

    rew = 0

    optimizerA.zero_grad()
    optimizerC.zero_grad()

    for t in range(episode_timout):
        action, log_prob = actor.act(state)
        value = critic.critique(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        rew += reward

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
        masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

        state = next_state

        if done:
            break 

    scores_deque.append(rew)
        
    next_value = critic.critique(next_state)
    returns = compute_returns(next_value, rewards, masks)

    log_probs = torch.cat(log_probs)
    returns = torch.cat(returns).detach()
    values = torch.cat(values)

    advantage = returns - values

    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    actor_loss.backward()
    critic_loss.backward()
    optimizerA.step()
    optimizerC.step()

    tot_score = np.sum(rewards)

    if iter % score_disp_interval == 0:
        print('Episode {}\tAverage Score: {:.2f}'.format(iter, np.mean(scores_deque)))

    if rew >= ss_score_threshold:
        horizon_ticker += 1
    else:
        horizon_ticker = 0

    if horizon_ticker > ss_horizon:
        ss_return_time = time.time()

end_time = time.time()

best_rew = 0
env_to_wrap = gym.make('CartPole-v1', render_mode="rgb_array")
env = record_video.RecordVideo(env_to_wrap, 'video')
for i in range(2):
    obs, done, rew = env.reset(), False, 0
    obs, _ = obs
    while (done != True) :
        action, _ =  actor.act(obs)
        obs, reward, terminate, truncate, info = env.step(action)
        done = terminate or truncate
        rew += reward
        sleep(0.01)
    print("episode : {}, reward : {}".format(i,rew)) 

    print(rew)
    best_rew = rew
    env.close()
    env_to_wrap.close()

final_train_time = end_time - start_time
threshld_reach_time = ss_return_time - start_time
episode_returns = best_rew/num_episodes

print("Train time: " + str(final_train_time) + " Time to reach " + str(ss_score_threshold) + " score: " + str(threshld_reach_time) + " ep returns: " + str(episode_returns))
