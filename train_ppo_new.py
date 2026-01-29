import random
import numpy as np
import gymnasium as gym
import os
from gymnasium.wrappers import AtariPreprocessing, FrameStack, TransformObservation

import torch
import argparse
from tqdm import tqdm
from ppo.agent import *
import sys
from ppo.neural_net import *
from collections import deque
# from ppo.ppo_utils import *
from ppo.ppo_trainer import *





if __name__ == '__main__':

    env = make_pong_env(render_mode='video',record_video=True)

    print("shape:",env.observation_space.shape)
    print("action:",env.action_space.n)

    # policy_net = PoliceNet(env.observation_space,env.action_space)
    # state,_ = env.reset()
    # # policy_net.print_shapes(state[0])

    ppo_train = PPOTrainer(env,batch_size=128,n_steps=2048)
    ppo_train.train(total_time_steps=1e6)

