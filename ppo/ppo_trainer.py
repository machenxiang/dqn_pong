import random
import numpy as np
import gymnasium as gym
import os
from gymnasium.wrappers import AtariPreprocessing, FrameStack, TransformObservation

import torch
import argparse
from tqdm import tqdm
from .agent import *
import sys
from .neural_net import *
from collections import deque
from .ppo_utils import *



class PPOTrainer:
    def __init__(self, env: gym.Env, batch_size: int = 128, n_steps: int = 2048):
        self.env = env
        self.batch_size = batch_size
        self.n_steps = n_steps
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
        self.agent = PPOAgent(env=env,actor_lr=1e-3,critic_lr=1e-2,lmbda=0.95,epochs=10,eps=0.2,gamma=0.98,device=device)

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
            "states":(self.states),
            "actions":(self.actions),
            "rewards":(self.rewards),
            "next_states":(self.next_states),
            "dones":(self.dones)
        }
    
    def collect_multi_trajectories(self,trajectory_num = 10):
        for _ in range(trajectory_num):
            state,_ = self.env.reset()
            for step in range(self.n_steps):
                action = self.agent.take_action(state)

                next_state,reward,terminated,truncated,info =self.env.step(action)
                done = terminated or truncated
    
                self.states.append(state)
                self.actions.append(action)
                self.rewards.append(reward)
                self.next_states.append(next_state)
                self.dones.append(done)
    
                state = next_state
                if done:
                    break

        print("type:",type(self.states[0]))
        print("size:",((self.states[0]).shape))
                
        return {
            "states":torch.stack(self.states).to(self.agent.device),
            "actions":torch.tensor(self.actions,dtype=torch.long, device=self.agent.device),
            "rewards":torch.tensor(self.rewards,dtype=torch.float32, device= self.agent.device),
            "next_states":torch.stack(self.next_states).to(self.agent.device),
            "dones":torch.tensor(self.dones ,dtype= torch.float32,device=self.agent.device)
        }       
            


    def clean_buffer(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def train_epoch(self,experience,n_epochs=10,mini_batch_size= 32):
        states=(experience['states'])
        actions = (experience['actions'])
        rewards = (experience['rewards'])
        next_states = (experience['next_states'])
        dones = (experience['dones'])

        print("hhtype:",type(states[0]))

        dataset_size = len(states)
        indices = torch.arange(dataset_size,device=self.agent.device)

        for epoch in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0,dataset_size,mini_batch_size):
                end = start+mini_batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions =actions[batch_indices]
                batch_rewards = rewards[batch_indices]
                batch_next_states = next_states[batch_indices]
                batch_dones = dones[batch_indices]
                self.agent.update(batch_states,batch_actions,batch_rewards,batch_next_states,batch_dones)



    def train(self,total_time_steps = 1e6):
        timestep = 0
        while timestep<total_time_steps:
            experiences = self.collect_multi_trajectories(trajectory_num=1)
            timestep+=self.n_steps
            self.train_epoch(experiences,n_epochs=10,mini_batch_size=32)

                
