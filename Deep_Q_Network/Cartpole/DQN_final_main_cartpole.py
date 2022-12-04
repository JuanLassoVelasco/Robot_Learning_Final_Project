import sys
import time
from time import sleep
import numpy as np
import gym
from gym.wrappers import record_video
from collections import deque

sys.path.insert(1, '/Users/juanlasso/Documents/Robot_Learning_Final_Project/Deep_Q_Network/')

from DQN_Agent import DQN_Agent

ss_horizon = 10
ss_score_threshold = 100

env = gym.make('CartPole-v1')
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
exp_storage_size = 256

learnRate = 1e-3
hid_layer_n = 64
score_disp_interval = 100

agent = DQN_Agent(seed = 1423, layer_sizes = [input_size, hid_layer_n, output_size], lr = learnRate, sync_freq = 5, exp_replay_size = exp_storage_size)

index = 0
for i in range(exp_storage_size):
    obs = env.reset()
    obs, _ = obs
    done = False
    while(done != True):
        action = agent.get_action(obs, env.action_space.n, epsilon=1)
        obs_next, reward, terminate, truncate, _ = env.step(action.item())
        done = terminate or truncate
        agent.collect_experience([obs, action.item(), reward, obs_next])
        obs = obs_next
        index += 1
        if( index > exp_storage_size ):
            break
            
losses_list, reward_list, episode_len_list, epsilon_list  = [], [], [], []
index = 128
episodes = 9000
episode_timout = 1000
epsilon = 1

scores_deque = deque(maxlen=100)

start_time = time.time()
ss_return_time = 0

horizon_ticker = 0

for i in range(episodes):
    obs, done, losses, ep_len, rew = env.reset(), False, 0, 0, 0
    obs, _ = obs
    for t in range(episode_timout):
        ep_len += 1 
        action = agent.get_action(obs, env.action_space.n, epsilon)
        obs_next, reward, terminate, truncate, _ = env.step(action.item())
        done = terminate or truncate
        agent.collect_experience([obs, action.item(), reward, obs_next])
       
        obs = obs_next
        rew  += reward
        index += 1
        
        if(index > 128):
            index = 0
            for j in range(4):
                loss = agent.train(batch_size=16)
                losses += loss    

        if done:
            break 

    scores_deque.append(rew)
          
    if epsilon > 0.05 :
        epsilon -= (1 / 5000)

    if i % score_disp_interval == 0:
        print('Episode {}\tAverage Score: {:.2f}'.format(i, np.mean(scores_deque)))
    
    losses_list.append(losses/ep_len), reward_list.append(rew), episode_len_list.append(ep_len), epsilon_list.append(epsilon)

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
        action =  agent.get_action(obs, env.action_space.n, epsilon = 0)
        obs, reward, terminate, truncate, info = env.step(action.item())
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
episode_returns = best_rew/episodes

print("Train time: " + str(final_train_time) + " Time to reach " + str(ss_score_threshold) + " score: " + str(threshld_reach_time) + " ep returns: " + str(episode_returns))
