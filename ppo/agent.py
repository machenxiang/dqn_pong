import gymnasium as gym  # 修改这里
from gymnasium import spaces  # 修改这里
import numpy as np
from .neural_net import *

import torch
import torch.nn.functional as F

class PPOAgent:
    def __init__(self,env:gym.Env,actor_lr,critic_lr,lmbda,epochs,eps,gamma,device):
        self.actor = PolicyNet(env.observation_space,env.action_space).to(device)
        self.critic = ValueNet(env.observation_space).to(device)
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
        state = state.to(self.device)
        with torch.no_grad():
            logits = self.actor(state)
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()

        return action.item()
    

    def compute_advantage(self,gamma,lmbda,td_delta):
        td_delta = td_delta.cpu().detach().numpy()
        advatage_list = []
        advatage = 0.0
        for delta in td_delta[::-1]:
            advatage = gamma*lmbda*advatage+delta
            advatage_list.append(advatage)
        advatage_list.reverse()

        return torch.tensor(advatage_list,dtype=torch.float32).to(self.device)
            
    
    def update(self,batch_states,batch_actions,batch_rewards,batch_next_states,batch_dones):
        states =  batch_states
        actions = batch_actions.long()
        rewards = batch_rewards
        next_states = batch_next_states
        dones = batch_dones


        # if actions.dim() == 1:
        #     actions = actions.unsqueeze(-1) 
        if rewards.dim() ==1:
            rewards = rewards.unsqueeze(-1)
        if dones.dim()==1:
            dones =dones.unsqueeze(-1)
        td_target = rewards+self.gamma*self.critic(next_states)*(1-dones)
        td_delta = td_target-self.critic(states)
        advantage = self.compute_advantage(self.gamma,self.lmbda,td_delta).detach()

        #old_log_probs = torch.log(self.actor(states).gather(1,actions)).detach()
        with torch.no_grad():
            logits = self.actor(states)
            dist = torch.distributions.Categorical(logits=logits)
            old_log_probs = dist.log_prob(actions)


        for _ in range(self.epochs):
            #log_probs = torch.log(self.actor(states).gather(1,actions))
            logits = self.actor(states)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            ratio = torch.exp(log_probs-old_log_probs)
            surr1 = ratio*advantage
            surr2 = torch.clamp(ratio,1-self.eps,1+self.eps)*advantage
            actor_loss = torch.mean(-torch.min(surr1,surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states),td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
        # return

    def save(self):
        torch.save({
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "actor_opt": actor_optimizer.state_dict(),
            "critic_opt": critic_optimizer.state_dict(),
            }, "ppo_checkpoint.pt")