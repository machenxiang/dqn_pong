import gymnasium as gym  # 修改这里
from gymnasium import spaces  # 修改这里
import numpy as np
from .neural_net import *

import torch
import torch.nn.functional as F

class PPOAgent:
    def __init__(self,env:gym.Env,actor_lr,critic_lr,lmbda,epochs,eps,gamma,device):
        self.actor = PoliceNet(env.observation_space,env.action_space)
        self.critic = ValueNet(env.observation_space)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr = actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr= critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device


    def take_action(self ,state):
        if state.dim() == 3:  # [C, H, W]
            state = state.unsqueeze(0)  # [1, C, H, W]
        # state = torch.tensor([state],dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    
    def update(self,batch_states,batch_actions,batch_rewards,batch_next_states,batch_dones):
        states =  batch_states
        actions = batch_actions
        rewards = batch_rewards
        next_states = batch_next_states
        dones = batch_dones

        print(dones)

        td_target = rewards+self.gamma*self.critic(next_states)*(1-dones)

        return
