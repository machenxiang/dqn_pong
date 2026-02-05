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
    def __init__(self, train_env: gym.Env,evalute_env:gym.Env, batch_size: int = 128, n_steps: int = 2048):
        self.train_env = train_env
        self.evalute_env =evalute_env
        self.batch_size = batch_size
        self.n_steps = n_steps
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
        self.agent = PPOAgent(env=self.train_env,actor_lr=3e-4,critic_lr=1e-3,lmbda=0.95,epochs=10,eps=0.2,gamma=0.98,device=device)

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []

    def collect_trajectories(self):
        state,_ = self.train_env.reset()
        episode_rewards = []
        for step in range(self.n_steps):
            action = self.agent.take_action(state)

            next_state,reward,terminated,truncated,info =self.train_env.step(action)
            done = terminated or truncated

            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)

            state = next_state
            if done:
                state = self.train_env.reset()
                break

        return {
            "states":(self.states),
            "actions":(self.actions),
            "rewards":(self.rewards),
            "next_states":(self.next_states),
            "dones":(self.dones)
        }
    
    def collect_multi_trajectories(self,trajectory_num = 10):
        self.clean_buffer()
        step_num = 0
        for _ in range(trajectory_num):
            state,_ = self.train_env.reset()
            done = False
            while not done:
                action = self.agent.take_action(state)

                next_state,reward,terminated,truncated,info =self.train_env.step(action)
                done = terminated or truncated
    
                self.states.append(state)
                self.actions.append(action)
                self.rewards.append(reward)
                self.next_states.append(next_state)
                self.dones.append(done)
    
                state = next_state
                step_num = step_num+1
                
        return {
            "states":torch.stack(self.states).to(self.agent.device),
            "actions":torch.tensor(self.actions,dtype=torch.long, device=self.agent.device),
            "rewards":torch.tensor(self.rewards,dtype=torch.float32, device= self.agent.device),
            "next_states":torch.stack(self.next_states).to(self.agent.device),
            "dones":torch.tensor(self.dones ,dtype= torch.float32,device=self.agent.device)
        },step_num       
            


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

        dataset_size = len(states)
        indices = torch.arange(dataset_size,device=self.agent.device)

        # print(f"dataset_size:{dataset_size},mini_batch_size:{mini_batch_size}")

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



    def train(self,total_time_steps = 1e6,eval_interval= 1000):
        timestep = 0
        episode = 0
        next_eval_timestep = eval_interval
        while timestep<total_time_steps:
            experiences,steps_collected = self.collect_multi_trajectories(trajectory_num=1)
            if experiences is None:
                print("no experience")
                continue
            timestep+=steps_collected
            episode+=1
            print(f"\nEpisode {episode}:")
            print(f"  收集步数: {steps_collected}")
            print(f"  总步数: {timestep}/{total_time_steps}")
            if len(experiences["states"])>=32:
                 self.train_epoch(experiences,n_epochs=10,mini_batch_size=32)

            if timestep>=next_eval_timestep:
                self.evaluate(n_episodes=5,record_video_episode=5,timestep=timestep)   
                self.agent.save()
                next_eval_timestep +=eval_interval



    def evaluate(self,n_episodes=5,record_video_episode=None ,timestep=0):
        total_rewards = []
        for episode in range(n_episodes):
            if episode ==record_video_episode-1:
                eval_env = make_pong_env(render_mode="rgb_array",record_video=True,timestep=timestep)
            else:
                eval_env = self.evalute_env

            state,_ = eval_env.reset()
            episode_reward = 0
            done = False

            while not done:
                with torch.no_grad():
                    action = self.agent.take_action(state)

                print("action:",action)
                next_state,reward,terminated,truncated,info =eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state

            total_rewards.append(episode_reward)

            if episode ==record_video_episode:
                eval_env.close()

        avg_reward = np.mean(episode_reward)
        print(f"评估: {n_episodes}回合平均奖励: {avg_reward:.2f}")
        return avg_reward

    