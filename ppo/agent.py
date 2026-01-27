import gymnasium as gym  # 修改这里
from gymnasium import spaces  # 修改这里
import numpy as np
from ppo.neural_net import *

import torch
import torch.nn.functional as F

class PPOAgent:
    def __init__(self,env:gym.Env,actor_lr,critic_lr,lmbda,epochs,eps,gamma,device):
        self.actor = PoliceNet(env.observation_space,env.action_space)
        self.critic = ValueNet(env.observation_space)
        self.actor_optimizer = torch.optim.adam(self.actor.parameters(),lr = actor_lr)
        self.critic_optimizer = torch.optim.adam(self.critic.parameters(),lr= critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device


    def take_action(self ,state):
        state = torch.tensor([state],dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    
    def update(self,transition_dict):
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],torch.float).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],dtype= torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).to(self.device)
        return
