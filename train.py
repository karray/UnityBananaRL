import os
import csv

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID' 
os.environ['CUDA_VISIBLE_DEVICES']='7'

from unityagents import UnityEnvironment
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from dqn_agent import Agent

env = UnityEnvironment(file_name="Banana_Linux_NoVis/Banana.x86_64")
# env = UnityEnvironment(file_name="VisualBanana_Linux/Banana.x86_64")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = state.shape[0]


writer = SummaryWriter('/home/aray/runs')

def train(agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]   
        score = 0
        done = False
        for t in range(max_t):
            action = agent.act(state, eps)

            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]                
            done = env_info.local_done[0]               
            
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        
        writer.add_scalar('score', score, i_episode)
        torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
        
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
#         if np.mean(scores_window)>=200.0:
            break
    return scores


agent = Agent(state_size, action_size)
scores = train(agent)