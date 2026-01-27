import random
import numpy as np
import gymnasium as gym
import os

from dqn.wrappers import *
import torch
import argparse
from tqdm import tqdm
from ppo.agent import PPO
import sys
from ppo.neural_net import *


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    # action = agent.take_action(state)
                    action = env.action_space.sample()
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    # next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                # agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO Atari')
    parser.add_argument('--load-checkpoint-file', type=str, default=None, 
                        help='Where checkpoint file should be loaded from (usually results/checkpoint.pth)')

    args = parser.parse_args()

    hyper_params = {
    "seed": 42,  # which seed to use
    "env": "PongNoFrameskip-v4",  # name of the game
    "replay-buffer-size": int(5e3),  # replay buffer size
    "learning-rate": 1e-4,  # learning rate for Adam optimizer
    "discount-factor": 0.99,  # discount factor
    "dqn_type": "neurips",
    # total number of steps to run the environment for
    "num-steps": int(1e6),
    "batch-size": 32,  # number of transitions to optimize at the same time
    "learning-starts": 10000,  # number of steps before learning starts
    "learning-freq": 1,  # number of iterations between every optimization step
    "num_episodes": 500,  # 
    "target-update-freq": 1000,  # number of iterations between every target network update
    "eps-start": 0.01,  # e-greedy start threshold
    "eps-end": 0.01,  # e-greedy end threshold
    "eps-fraction": 0.1,  # fraction of num-steps
    "print-freq": 10
    }

    np.random.seed(hyper_params["seed"])
    random.seed(hyper_params["seed"])

    assert "NoFrameskip" in hyper_params["env"], "Require environment with no frameskip"
    
    # 创建视频目录
    video_dir = './ppo_video/'
    os.makedirs(video_dir, exist_ok=True)
    
    # 使用Gymnasium API创建环境
    env = gym.make(hyper_params["env"], render_mode='rgb_array')
    env.reset(seed=hyper_params["seed"])

    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    
    # FireResetEnv在reset时需要返回两个值，但在wrapper中可能有问题
    # 先暂时注释掉，或者修复FireResetEnv
    env = FireResetEnv(env)
    
    env = WarpFrame(env)
    env = PyTorchFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    
    # 使用Gymnasium的RecordVideo
    env = gym.wrappers.RecordVideo(
        env, 
        video_dir,
        episode_trigger=lambda episode_id: episode_id % 50 == 0,
    )

    print("shape:",env.observation_space.shape)
    print("action:",env.action_space.n)

    # policy_net = PoliceNet(env.observation_space,env.action_space)
    state,_ = env.reset()
    print("state type:",type(state))
    print(state.array)
    sys.exit(0)
    agent = PPO
    train_on_policy_agent(env,agent,hyper_params["num_episodes"])