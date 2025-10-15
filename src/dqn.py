import gymnasium as gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ==============================================================================
# 0. 超参数与设置
# ==============================================================================
# 如果有GPU，则使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay Buffer 中存储的 transition 对象
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# 超参数
BATCH_SIZE = 128  # 每次从 Replay Buffer 中采样的数量
GAMMA = 0.99  # 折扣因子
EPS_START = 0.9  # Epsilon 的初始值 (高探索率)
EPS_END = 0.05  # Epsilon 的最终值 (低探索率)
EPS_DECAY = 1000  # Epsilon 的衰减速率
TAU = 0.005  # 目标网络软更新的系数
LR = 1e-4  # 优化器的学习率
BUFFER_SIZE = 10000  # Replay Buffer 的大小


# ==============================================================================
# 1. Q-Network (神经网络)
# ==============================================================================
class QNetwork(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# ==============================================================================
# 2. Replay Buffer (经验回放池)
# ==============================================================================
class ReplayBuffer:
    def __init__(self, capacity):
        # 使用 deque 实现一个固定大小的队列，当队列满时，旧的经验会被自动丢弃
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """保存一个 transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """从 memory 中随机采样一个 batch 的 transitions"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# ==============================================================================
# 3. DQN Agent (集成了网络、经验池和算法逻辑)
# ==============================================================================
class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # 创建主网络和目标网络
        self.policy_net = QNetwork(self.state_dim, self.action_dim).to(device)
        self.target_net = QNetwork(self.state_dim, self.action_dim).to(device)
        # 初始化时，将目标网络的权重设置为与主网络相同
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络不进行训练

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.steps_done = 0

    def select_action(self, state):
        """使用 Epsilon-Greedy 策略选择动作"""
        sample = random.random()
        # 计算当前 Epsilon 值
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            # 利用 (Exploitation): 选择Q值最高的动作
            with torch.no_grad():
                # state 需要增加一个 batch 维度 [C, H, W] -> [1, C, H, W]
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # 探索 (Exploration): 随机选择一个动作
            return torch.tensor([[self.env.action_space.sample()]], device=device, dtype=torch.long)

    def update_model(self):
        """
        核心学习步骤：从经验池采样，计算损失，并更新网络。
        这是最关键的部分，也是理论转化为代码的地方。
        """
        # 如果经验池中的样本数量不足一个 batch, 则不进行学习
        if len(self.memory) < BATCH_SIZE:
            return

        # 1. 从经验池中采样
        transitions = self.memory.sample(BATCH_SIZE)
        # 将 batch of transitions 转换为 a transition of batches
        # (例如, 将 N 个 (s,a,s',r) 元组, 转换为 (s_1,...,s_N), (a_1,...,a_N), ...)
        batch = Transition(*zip(*transitions))

        # 2. 准备数据
        # 创建一个掩码，用于标记非最终状态 (即游戏未结束的状态)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        # 提取所有非最终状态的 next_state
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # 3. 计算 Q(s_t, a_t)
        #    这是主网络 (policy_net) 预测的Q值。
        #    我们只关心实际采取的那个动作 a_t 的Q值。
        #    .gather(1, action_batch) 的作用就是从所有动作的Q值中，选出 action_batch 指定的那个。
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # 4. 计算 V(s_{t+1}) = max_{a'} Q_target(s_{t+1}, a')
        #    这是目标网络 (target_net) 预测的下一状态的最大Q值。
        #    对于所有最终状态，其后续价值为0。
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        # 5. 计算期望的 Q 值 (TD Target)
        #    Expected Q Value = r + γ * V(s_{t+1})
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # 6. 计算损失
        #    使用 Smooth L1 Loss (Huber Loss)，它比 MSELoss 更稳定。
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # 7. 优化模型 (反向传播)
        self.optimizer.zero_grad()
        loss.backward()
        # 对梯度进行裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_net(self):
        """
        软更新目标网络的权重:
        θ_target = τ*θ_local + (1 - τ)*θ_target
        这是一种比直接复制更平滑的更新方式。
        """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        self.target_net.load_state_dict(target_net_state_dict)


# ==============================================================================
# 4. 训练主循环
# ==============================================================================
if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    agent = DQNAgent(env)
    episode_durations = []

    num_episodes = 600
    for i_episode in range(num_episodes):
        # 初始化环境和状态
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        total_reward = 0
        done = False
        while not done:
            # 选择并执行动作
            action = agent.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # 在 memory 中存储 transition
            agent.memory.push(state, action, next_state, reward)

            # 移至下一个状态
            state = next_state

            # 执行一步优化 (在 policy network 上)
            agent.update_model()

            # 软更新目标网络的权重
            agent.update_target_net()

        episode_durations.append(total_reward)
        if i_episode % 20 == 0:
            print(f"Episode {i_episode}, Avg Reward (last 20): {np.mean(episode_durations[-20:]):.2f}")

    print('训练完成')
    env.close()

    # 绘制结果
    plt.figure(1)
    plt.title('Result')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(episode_durations)
    plt.show()