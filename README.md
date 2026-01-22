# DQN-Atari
Deep Q-network implementation for [Pong-vo](https://gym.openai.com/envs/Pong-v0/).  The implementation follows from the paper - [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) and [Human-level control through deep reinforcement
learning](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf).
## Results
### Video of Gameplay - DQN Nature Paper
[![DQN Video](http://img.youtube.com/vi/DcyMFIKsVNI/0.jpg)](http://www.youtube.com/watch?v=DcyMFIKsVNI "DQN For Atari Pong")
### Reward per Episode
![Rewards Per Episode](./results/results_per_episode.png)


## Summary of Implementation
### DQN Nature Architecture Implementation
- Input : 84 × 84 × 4 image (using the last 4 frames of a history)
- Conv Layer 1 : 32 8 × 8 filters with stride 4
- Conv Layer 2: 64 4 × 4 filters with stride 2
- Conv Layer 3: 64 3 × 3 filters with stride 1
- Fully Connected 1 : fully-connected and consists of 256 rectifier units
- Output : fully connected linear layer with a single output for each valid action.

### DQN Neurips Architecture Implementation
- Input : 84 × 84 × 4 image (using the last 4 frames of a history)
- Conv Layer 1 : 16 8 × 8 filters with stride 4
- Conv Layer 2: 32 4 × 4 filters with stride 2
- Fully Connected 1 : fully-connected and consists of 256 rectifier units
- Output : fully connected linear layer with a single output for each valid action.
#### Other Params
- Optimizer: RMSProp 
- Batch Size:  32
- E-greedy : 0.1


## How to run
### Create a new environment
Example: 
```
conda create -n dqn_pong
``` 

### Install Dependencies
```
pip install -r requirements.txt
```
### To use `gym.wrappers.Monitor` to record the last episode
```
sudo apt-get install ffmpeg
```

### Run Training from Scratch
```
python train_atari.py
```
### Use a trained agent
```
python train_atari.py --load-checkpoint-file results/checkpoint_dqn_nature.pth
```
## View Progress
A video is recorded every 50 episodes. See videos in `/video/` folder. 

## reset work flow
1. RecordVideo.reset() 开始 [最外层]
   │
   ├─ 打印视频信息
   │
   └─ 调用 super().reset() → FrameStack.reset()
       │
       ├─ FrameStack.reset() 开始
       │   │
       │   ├─ 调用 self.env.reset() → ClipRewardEnv.reset()
       │   │   │
       │   │   ├─ ClipRewardEnv.reset() [RewardWrapper基类]
       │   │   │   │
       │   │   │   └─ 调用 self.env.reset() → PyTorchFrame.reset()
       │   │   │       │
       │   │   │       ├─ PyTorchFrame.reset() [ObservationWrapper基类]
       │   │   │       │   │
       │   │   │       │   └─ 调用 self.env.reset() → WarpFrame.reset()
       │   │   │       │       │
       │   │   │       │       ├─ WarpFrame.reset() [ObservationWrapper基类]
       │   │   │       │       │   │
       │   │   │       │       │   └─ 调用 self.env.reset() → FireResetEnv.reset()
       │   │   │       │       │       │
       │   │   │       │       │       ├─ FireResetEnv.reset() 开始
       │   │   │       │       │       │   │
       │   │   │       │       │       │   ├─ 打印: "self type: <class 'dqn.wrappers.FireResetEnv'>"
       │   │   │       │       │       │   │
       │   │   │       │       │       │   ├─ 调用 self.env.reset() → EpisodicLifeEnv.reset()
       │   │   │       │       │       │   │   │
       │   │   │       │       │       │   │   ├─ EpisodicLifeEnv.reset() 开始
       │   │   │       │       │       │   │   │   │
       │   │   │       │       │       │   │   │   ├─ 检查 self.was_real_done
       │   │   │       │       │       │   │   │   │
       │   │   │       │       │       │   │   │   ├─ 调用 self.env.reset() → MaxAndSkipEnv.reset()
       │   │   │       │       │       │   │   │   │   │
       │   │   │       │       │       │   │   │   │   ├─ MaxAndSkipEnv.reset()
       │   │   │       │       │       │   │   │   │   │   │
       │   │   │       │       │       │   │   │   │   │   └─ 调用 self.env.reset() → NoopResetEnv.reset()
       │   │   │       │       │       │   │   │   │   │       │
       │   │   │       │       │       │   │   │   │   │       ├─ NoopResetEnv.reset() 开始
       │   │   │       │       │       │   │   │   │   │       │   │
       │   │   │       │       │       │   │   │   │   │       │   ├─ 调用 self.env.reset() → 原始环境.reset()
       │   │   │       │       │       │   │   │   │   │       │   │   │
       │   │   │       │       │       │   │   │   │   │       │   │   └─ 原始Atari环境真正重置
       │   │   │       │       │       │   │   │   │   │       │   │       ↑ 返回初始观察值
       │   │   │       │       │       │   │   │   │   │       │   ├─ 执行随机noop操作
       │   │   │       │       │       │   │   │   │   │       │   │   │
       │   │   │       │       │       │   │   │   │   │       │   │   ├─ 循环执行随机次数的动作0
       │   │   │       │       │       │   │   │   │   │       │   │   │   │
       │   │   │       │       │       │   │   │   │   │       │   │   │   └─ 每次调用: self.env.step(0)
       │   │   │       │       │       │   │   │   │   │       │   │   │       │
       │   │   │       │       │       │   │   │   │   │       │   │   │       └─ 进入step调用链...
       │   │   │       │       │       │   │   │   │   │       │   │   │
       │   │   │       │       │       │   │   │   │   │       │   │   └─ 结束noop循环
       │   │   │       │       │       │   │   │   │   │       │   │
       │   │   │       │       │       │   │   │   │   │       │   └─ 返回观察值
       │   │   │       │       │       │   │   │   │   │       │
       │   │   │       │       │       │   │   │   │   │       └─ MaxAndSkipEnv收到观察值
       │   │   │       │       │       │   │   │   │   │           │
       │   │   │       │       │       │   │   │   │   │           └─ 直接返回观察值（无处理）
       │   │   │       │       │       │   │   │   │   │
       │   │   │       │       │       │   │   │   │   └─ EpisodicLifeEnv收到观察值
       │   │   │       │       │       │   │   │   │       │
       │   │   │       │       │       │   │   │   │       ├─ 记录 self.lives
       │   │   │       │       │       │   │   │   │       │
       │   │   │       │       │       │   │   │   │       └─ 返回观察值
       │   │   │       │       │       │   │   │   │
       │   │   │       │       │       │   │   │   └─ FireResetEnv收到观察值
       │   │   │       │       │       │   │   │       │
       │   │   │       │       │       │   │   │       ├─ 执行 self.env.step(1) [FIRE动作]
       │   │   │       │       │       │   │   │       │   │
       │   │   │       │       │       │   │   │       │   └─ 进入step调用链（详细见下方）
       │   │   │       │       │       │   │   │       │
       │   │   │       │       │target="_blank"│   │   │       ├─ 检查游戏是否结束，可能再次reset
       │   │   │       │       │       │   │   │       │
       │   │   │       │       │       │   │   │       ├─ 执行 self.env.step(2) [RIGHT动作]
       │   │   │       │       │       │   │   │       │   │
       │   │   │       │       │       │   │   │       │   └─ 进入step调用链
       │   │   │       │       │       │   │   │       │
       │   │   │       │       │       │   │   │       ├─ 检查游戏是否结束，可能再次reset
       │   │   │       │       │       │   │   │       │
       │   │   │       │       │       │   │   │       └─ 返回最终观察值
       │   │   │       │       │       │   │   │
       │   │   │       │       │       │   │   └─ WarpFrame收到观察值
       │   │   │       │       │       │   │       │
       │   │   │       │       │       │   │       ├─ 调用 observation() 方法：
       │   │   │       │       │       │   │       │   - 转换为灰度图
       │   │   │       │       │       │   │       │   - 调整大小为84×84
       │   │   │       │       │       │   │       │
       │   │   │       │       │       │   │       └─ 返回处理后的观察值
       │   │   │       │       │       │   │
       │   │   │       │       │       │   └─ PyTorchFrame收到观察值
       │   │   │       │       │       │       │
       │   │   │       │       │       │       ├─ 调用 observation() 方法：
       │   │   │       │       │       │       │   - 调整维度：HWC → CHW
       │   │   │       │       │       │       │
       │   │   │       │       │       │       └─ 返回处理后的观察值
       │   │   │       │       │       │
       │   │   │       │       │       └─ ClipRewardEnv收到观察值
       │   │   │       │       │           │
       │   │   │       │       │           └─ 返回观察值（奖励在step中处理）
       │   │   │       │       │
       │   │   │       │       └─ FrameStack收到观察值
       │   │   │       │           │
       │   │   │       │           ├─ 将观察值重复k次添加到frames队列
       │   │   │       │           │
       │   │   │       │           ├─ 调用 _get_ob() 获取堆叠帧
       │   │   │       │           │
       │   │   │       │           └─ 返回 LazyFrames 对象
       │   │   │       │
       │   │   │       └─ RecordVideo收到观察值
       │   │   │           │
       │   │   │           ├─ 录制第一帧视频
       │   │   │           │
       │   │   │           └─ 返回最终观察值给用户
       │   │   │
       └─ 用户得到 state = LazyFrames对象, info = 信息字典


## step workflow
1. RecordVideo.step(action) 开始 [最外层]
   │
   ├─ 可能录制视频帧
   │
   └─ 调用 super().step(action) → FrameStack.step(action)
       │
       ├─ FrameStack.step(action) 开始
       │   │
       │   ├─ 调用 self.env.step(action) → ClipRewardEnv.step(action)
       │   │   │
       │   │   ├─ ClipRewardEnv.step(action) [RewardWrapper基类]
       │   │   │   │
       │   │   │   ├─ 调用 self.env.step(action) → PyTorchFrame.step(action)
       │   │   │   │   │
       │   │   │   │   ├─ PyTorchFrame.step(action) [ObservationWrapper基类]
       │   │   │   │   │   │
       │   │   │   │   │   ├─ 调用 self.env.step(action) → WarpFrame.step(action)
       │   │   │   │   │   │   │
       │   │   │   │   │   │   ├─ WarpFrame.step(action) [ObservationWrapper基类]
       │   │   │   │   │   │   │   │
       │   │   │   │   │   │   │   ├─ 调用 self.env.step(action) → FireResetEnv.step(action)
       │   │   │   │   │   │   │   │   │
       │   │   │   │   │   │   │   │   ├─ FireResetEnv.step(action) 开始
       │   │   │   │   │   │   │   │   │   │
       │   │   │   │   │   │   │   │   │   └─ 调用 self.env.step(action) → EpisodicLifeEnv.step(action)
       │   │   │   │   │   │   │   │   │       │
       │   │   │   │   │   │   │   │   │       ├─ EpisodicLifeEnv.step(action) 开始
       │   │   │   │   │   │   │   │   │       │   │
       │   │   │   │   │   │   │   │   │       │   ├─ 调用 self.env.step(action) → MaxAndSkipEnv.step(action)
       │   │   │   │   │   │   │   │   │       │   │   │
       │   │   │   │   │   │   │   │   │       │   │   ├─ MaxAndSkipEnv.step(action) 开始
       │   │   │   │   │   │   │   │   │       │   │   │   │
       │   │   │   │   │   │   │   │   │       │   │   │   ├─ 循环执行 skip=4 次相同动作
       │   │   │   │   │   │   │   │   │       │   │   │   │   │
       │   │   │   │   │   │   │   │   │       │   │   │   │   ├─ 每次迭代：
       │   │   │   │   │   │   │   │   │       │   │   │   │   │   │
       │   │   │   │   │   │   │   │   │       │   │   │   │   │   └─ 调用 self.env.step(action) → NoopResetEnv.step(action)
       │   │   │   │   │   │   │   │   │       │   │   │   │   │       │
       │   │   │   │   │   │   │   │   │       │   │   │   │   │       ├─ NoopResetEnv.step(action) [直接调用父类]
       │   │   │   │   │   │   │   │   │       │   │   │   │   │       │   │
       │   │   │   │   │   │   │   │   │       │   │   │   │   │       │   └─ 调用 self.env.step(action) → 原始环境.step(action)
       │   │   │       │       │       │   │   │   │   │   │   │   │   │   │   │
       │   │   │       │       │       │   │   │   │   │   │   │   │   │   │   └─ 原始Atari环境执行动作
       │   │   │       │       │       │   │   │   │   │   │   │   │   │   │       ↑ 返回原始结果
       │   │   │       │       │       │   │   │   │   │   │   │   │   │   │
       │   │   │       │       │       │   │   │   │   │   │   │   │   │   └─ NoopResetEnv返回结果
       │   │   │       │       │       │   │   │   │   │   │   │   │   │
       │   │   │       │       │       │   │   │   │   │   │   │   │   └─ MaxAndSkipEnv累加奖励
       │   │   │       │       │       │   │   │   │   │   │   │   │
       │   │   │       │       │       │   │   │   │   │   │   │   └─ 结束循环
       │   │   │       │       │       │   │   │   │   │   │   │
       │   │   │       │       │       │   │   │   │   │   │   ├─ 取最后两帧的最大值
       │   │   │       │       │       │   │   │   │   │   │   │
       │   │   │       │       │       │   │   │   │   │   │   └─ 返回处理后的帧和总奖励
       │   │   │       │       │       │   │   │   │   │   │
       │   │   │       │       │       │   │   │   │   │   └─ EpisodicLifeEnv收到结果
       │   │   │       │       │       │   │   │   │   │       │
       │   │   │       │       │       │   │   │   │   │       ├─ 检查生命值变化
       │   │   │       │       │       │   │   │   │   │       │   - 如果生命减少，设置 terminated=True
       │   │   │       │       │       │   │   │   │   │       │
       │   │   │       │       │       │   │   │   │   │       ├─ 更新 self.lives
       │   │   │       │       │       │   │   │   │   │       │
       │   │   │       │       │       │   │   │   │   │       └─ 返回处理后的结果
       │   │   │       │       │       │   │   │   │   │
       │   │   │       │       │       │   │   │   │   └─ FireResetEnv收到结果
       │   │   │       │       │       │   │   │   │       │
       │   │   │       │       │       │   │   │   │       └─ 直接返回结果（无处理）
       │   │   │       │       │       │   │   │   │
       │   │   │       │       │       │   │   │   └─ WarpFrame收到结果
       │   │   │       │       │       │   │   │       │
       │   │   │       │       │       │   │   │       ├─ 调用 observation() 处理观察值
       │   │   │       │       │       │   │   │       │   - 转换为灰度图
       │   │   │       │       │       │   │   │       │   - 调整大小为84×84
       │   │   │       │       │       │   │   │       │
       │   │   │       │       │       │   │   │       └─ 返回处理后的结果
       │   │   │       │       │       │   │   │
       │   │   │       │       │       │   │   └─ PyTorchFrame收到结果
       │   │   │       │       │       │   │       │
       │   │   │       │       │       │   │       ├─ 调用 observation() 处理观察值
       │   │   │       │       │       │   │       │   - 调整维度：HWC → CHW
       │   │   │       │       │       │   │       │
       │   │   │       │       │       │   │       └─ 返回处理后的结果
       │   │   │       │       │       │   │
       │   │   │       │       │       │   └─ ClipRewardEnv收到结果
       │   │   │       │       │       │       │
       │   │   │       │       │       │       ├─ 调用 reward() 方法处理奖励
       │   │   │       │       │       │       │   - 裁剪奖励为[-1, 0, 1]
       │   │   │       │       │       │       │
       │   │   │       │       │       │       └─ 返回处理后的结果
       │   │   │       │       │       │
       │   │   │       │       │       └─ FrameStack收到结果
       │   │   │       │       │           │
       │   │   │       │       │           ├─ 将新观察值添加到frames队列
       │   │   │       │       │           │
       │   │   │       │       │           ├─ 调用 _get_ob() 获取新的堆叠帧
       │   │   │       │       │           │
       │   │   │       │       │           └─ 返回处理后的结果
       │   │   │       │       │
       │   │   │       │       └─ RecordVideo收到结果
       │   │   │       │           │
       │   │   │       │           ├─ 可能录制视频帧
       │   │   │       │           │
       │   │   │       │           └─ 返回最终结果给用户
       │   │   │       │
       └─ 用户得到: next_state, reward, terminated, truncated, info