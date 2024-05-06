import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 定义环境
class NoahEnv(gym.Env):
    def __init__(self):
        super(NoahEnv, self).__init__()
        self.cpu_usage = 0.5  # 初始化CPU使用率
        self.request_rate = 0.8  # 请求完成率
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32)
        self.max_steps = 1e7  # 设定最大步骤数
        self.current_step = 0

    def step(self, action):
        self.current_step += 1
        done = False
        reward = 0
        # 模拟环境动作反应
        self.cpu_usage += action[0] * 0.1 - 0.05  # 动作可能导致CPU使用率变化
        self.request_rate += action[0] * 0.1 - 0.02  # 同时影响请求完成率

        if self.cpu_usage > 0.8:
            reward -= 20  # CPU使用率过高，给予负奖励
        if self.request_rate > 0.95:
            reward += 10  # 高请求完成率，给予正奖励
        elif self.request_rate < 0.7:
            reward -= 10  # 请求完成率低，负奖励

        # 检查是否结束
        if self.cpu_usage > 1.0 or self.current_step >= self.max_steps:
            done = True
            
        # 更新状态
        next_state = [self.cpu_usage, self.request_rate]
        return np.array(next_state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.cpu_usage = 0.5
        self.request_rate = 0.8
        self.current_step = 0
        return np.array([self.cpu_usage, self.request_rate], dtype=np.float32)

# 使用RL训练
env = NoahEnv()
state = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # 随机选择一个动作
    next_state, reward, done, _ = env.step(action)  # 执行动作
    print(f"Step: {env.current_step}, State: {state}, Action: {action}, Reward: {reward}")
    state = next_state