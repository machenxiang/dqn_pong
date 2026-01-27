import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym  # 修改这里
from gymnasium import spaces  # 修改这里

class PoliceNet(torch.nn.Module):
    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Discrete):
        super().__init__()
        assert type(
            observation_space)==spaces.Box, 'observation_space must be of type Box'
        assert len(
            observation_space.shape) == 3, 'observation space must have the form channels x width x height'
        assert type(
            action_space) == spaces.Discrete, 'action_space must be of type Discrete'
        
        self.conv= torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=observation_space.shape[0],out_channels=16,kernel_size=8,stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=16,out_channels=32,kernel_size=4,stride=2),
            torch.nn.ReLU()
        )
        self.fc=torch.nn.Sequential(
            torch.nn.Linear(in_features=32*9*9,out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256,out_features=action_space.n)
        )

    def forward(self,x):
        conv_out=self.conv(x).view(x.size()[0],-1)
        return F.softmax(self.fc(conv_out))
    
    def print_shapes(self, x):
        print(f"输入形状: {x.shape}")  

        conv1_out = self.conv[0](x)
        print("conv 层数：",len( self.conv))
        print(f"Conv1后: {conv1_out.shape}")  

        conv2_out = self.conv[2](conv1_out)
        print(f"Conv2后: {conv2_out.shape}")  

        flattened = conv2_out.view(x.size()[0], -1)
        print(f"展平后: {flattened.shape}")  # [batch, ?]


class ValueNet(torch.nn.Module):
    def __init__(self,
                 observation_space: spaces.Box):
        super().__init__()
        self.conv = torch.nn.Sequential( 
            torch.nn.Conv2d(in_channels=observation_space.shape[0],out_channels=16,kernel_size=8,stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=16,out_channels=32,kernel_size=4,stride=2),
            torch.nn.ReLU()
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=32*9*9,out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256,out_features=1)
        )

    def forward(self,x):
        x= F.relu(self.conv(x))
        return self.fc(x)




