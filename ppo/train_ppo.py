import random
import numpy as np
import gymnasium as gym
import os
from gymnasium.wrappers import AtariPreprocessing, FrameStack, TransformObservation

import torch
import argparse
from tqdm import tqdm
from ppo.agent import PPO
import sys
from ppo.neural_net import *
from collections import deque
from ppo.ppo_utils import *



class PPOTrainer:
    def __init__(self, env: gym.Env, batch_size: int = 128, n_steps: int = 2048):
        self.env = env
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.agent = PPOAgent

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []

    def collect_trajectories(self):
        state,_ = self.env.reset()
        episode_rewards = []
        for step in range(self.n_steps):
            action = self.agent.take_action(state)

            next_state,reward,terminated,truncated,info =env.step(action)
            done = terminated or truncated

            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)

            state = next_state
            if done:
                state = self.env.reset()
                break

        return {
            "states":np.array(self.states),
            "actions":np.array(self.actions),
            "rewards":np.array(self.rewards),
            "next_states":np.array(self.next_states),
            "dones":np.array(self.dones)
        }
    
    def collect_multi_trajectories(self,trajectory_num = 10):
        for _ in range(trajectory_num):
            state,_ = self.env.reset()
            for step in range(self.n_steps):
                action = self.agent.take_action(state)

                next_state,reward,terminated,truncated,info =env.step(action)
                done = terminated or truncated
    
                self.states.append(state)
                self.actions.append(action)
                self.rewards.append(reward)
                self.next_states.append(next_state)
                self.dones.append(done)
    
                state = next_state
                if done:
                    break
                
        return {
            "states":np.array(self.states),
            "actions":np.array(self.actions),
            "rewards":np.array(self.rewards),
            "next_states":np.array(self.next_states),
            "dones":np.array(self.dones)
        }       
            


    def clean_buffer(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def train_epoch(experience,n_epochs=10,mini_batch_size= 32):
        states=torch.FloatTensor(experience['state'])
        actions = torch.FloatTensor(experience['actions'])
        rewards = torch.FloatTensor(experience['rewards'])
        next_states = torch.FloatTensor(experience['next_states'])
        dones = torch.FloatTensor(experience['dones'])

        dataset_size = len(states)
        indices = np.arange(dataset_size)

        for epoch in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0,dataset_size,mini_batch_size):
                end = start+mini_batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                ba



    def train(self,total_time_steps = 1e6):
        timestep = 0
        while timestep<total_time_steps:
            experience = self.collect_trajectories()
            timestep+=self.n_steps
