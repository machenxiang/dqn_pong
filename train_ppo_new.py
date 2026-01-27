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





if __name__ == '__main__':

    env = make_pong_env(render_mode='video',record_video=True)

    print("shape:",env.observation_space.shape)
    print("action:",env.action_space.n)

    policy_net = PoliceNet(env.observation_space,env.action_space)
    state,_ = env.reset()
    # policy_net.print_shapes(state[0])


    sys.exit(0)

    agent = PPO
    train_on_policy_agent(env,agent,100)