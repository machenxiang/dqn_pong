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

def make_pong_env(render_mode=None, record_video=False):
    """
    简化版本：让FrameStack输出(4, 84, 84)格式
    """
    # 创建基础环境
    env = gym.make(
        "PongNoFrameskip-v4",
        render_mode='rgb_array' if render_mode else None,
        frameskip=1,  # 使用AtariPreprocessing的frame_skip
        repeat_action_probability=0.0,
        full_action_space=False  # Pong只需要6个动作
    )
    
    # Atari预处理 - 不添加通道维度
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=True,
        grayscale_obs=True,
        grayscale_newaxis=False,  # ❗ 不添加通道维度
        scale_obs=True,
    )
    # 现在obs形状为 (84, 84)
    
    # 自定义FrameStack包装器，返回(4, 84, 84)数组
    env = CustomFrameStack(env, num_stack=4)
    
    # 转换为PyTorch张量
    env = PyTorchTensorWrapper(env)
    
    # 录制视频
    if record_video:
        video_dir = './ppo_videos/'
        os.makedirs(video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_dir,
            episode_trigger=lambda episode_id: episode_id % 50 == 0,
        )
    
    return env


class CustomFrameStack(gym.Wrapper):
    """
    自定义帧堆叠，直接返回(4, 84, 84)的numpy数组
    """
    
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        
        # 更新观测空间
        obs_shape = env.observation_space.shape  # (84, 84)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_stack, *obs_shape),  # (4, 84, 84)
            dtype=np.float32
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # 用第一帧填充帧栈
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        """返回堆叠的帧作为numpy数组"""
        # 确保有足够的帧
        while len(self.frames) < self.num_stack:
            self.frames.append(self.frames[-1] if self.frames else np.zeros((84, 84)))
        
        # 堆叠帧
        stacked = np.stack(list(self.frames), axis=0)  # (4, 84, 84)
        return stacked.astype(np.float32)


class PyTorchTensorWrapper(gym.ObservationWrapper):
    """简单转换为PyTorch张量"""
    
    def __init__(self, env):
        super().__init__(env)
        # 保持原始观测空间，只是转换类型
    
    def observation(self, observation):
        if isinstance(observation, torch.Tensor):
            return observation
        return torch.FloatTensor(observation)
