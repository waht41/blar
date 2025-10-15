import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import random
from collections import deque

# --- 超参数 (Hyperparameters) ---
GAMMA = 0.99  # 折扣因子
LEARNING_RATE = 0.001  # 学习率
BUFFER_SIZE = 10000  # 经验回放池大小
BATCH_SIZE = 64  # 批处理大小
TARGET_UPDATE_FREQUENCY = 100  # 目标网络更新频率 (每 C 步)
EPSILON_START = 1.0  # Epsilon-greedy 策略的起始探索率
EPSILON_END = 0.01  # Epsilon-greedy 策略的最终探索率
EPSILON_DECAY = 500  # Epsilon 的衰减率


# --- 1. Q-Network 定义 ---
# 一个简单的多层感知机 (MLP)
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# --- 2. 经验回放池 ---
class ReplayBuffer:
    def __init__(self, capacity):
        # 使用 deque 实现一个固定大小的队列
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """将一个经验元组存入 buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """从 buffer 中随机采样一个批次的经验"""
        # 从 buffer 中随机选择 batch_size 个样本
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        # 将样本转换为 numpy array
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        """返回当前 buffer 中的经验数量"""
        return len(self.buffer)


# --- 3. DDQN Agent ---
class DDQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 初始化评估网络和目标网络
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_q_network = QNetwork(state_dim, action_dim)

        # 将目标网络的权重初始化为与评估网络相同
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)

        # 经验回放池
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)

        # 用于 epsilon-greedy 策略的步数计数器
        self.steps_done = 0

    def select_action(self, state):
        """使用 Epsilon-Greedy 策略选择动作"""
        # 计算当前的 epsilon 值
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                  np.exp(-1. * self.steps_done / EPSILON_DECAY)
        self.steps_done += 1

        if random.random() > epsilon:
            # 利用 (Exploitation)
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                # 从 Q-network 获取对每个动作的 Q 值
                q_values = self.q_network(state)
                # 选择 Q 值最大的动作
                action = q_values.max(1)[1].item()
        else:
            # 探索 (Exploration)
            action = random.randrange(self.action_dim)
        return action

    def learn(self):
        """从经验回放池中采样数据并更新网络"""
        if len(self.replay_buffer) < BATCH_SIZE:
            return  # 如果 buffer 中的数据不够一个 batch，则不学习

        # 1. 从 Replay Buffer 中采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        # 2. 将数据转换为 PyTorch Tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)  # [B] -> [B, 1]
        rewards = torch.FloatTensor(rewards).unsqueeze(1)  # [B] -> [B, 1]
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)  # [B] -> [B, 1]

        # --- DDQN 核心逻辑 ---
        # 3. 计算当前状态的 Q 值 (Q(s, a))
        # self.q_network(states) -> [B, action_dim]
        # .gather(1, actions) -> 根据 actions 的索引, 在 dim=1 上选取对应的 Q 值
        current_q_values = self.q_network(states).gather(1, actions)

        # 4. 计算下一个状态的 Q 值 (目标 Q 值)
        with torch.no_grad():  # 目标 Q 值的计算不涉及梯度
            # 4.1. 使用 *评估网络* 选择在 next_state 时的最优动作 a'
            # .max(1)[1] 返回最大值的索引 (即动作)
            # .unsqueeze(1) 调整形状为 [B, 1]
            next_actions = self.q_network(next_states).max(1)[1].unsqueeze(1)

            # 4.2. 使用 *目标网络* 计算该动作 a' 对应的 Q 值: Q_target(s', a')
            # 这就是 DDQN 与 DQN 的关键区别！
            next_q_values = self.target_q_network(next_states).gather(1, next_actions)

            # 4.3. 计算 TD Target
            # 如果 done=True (游戏结束), 则目标 Q 值就是 reward
            target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

        # 5. 计算损失函数 (MSE Loss 或 Smooth L1 Loss)
        loss = F.mse_loss(current_q_values, target_q_values)

        # 6. 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """定期将评估网络的权重复制到目标网络"""
        self.target_q_network.load_state_dict(self.q_network.state_dict())


# --- 伪代码/主训练循环 ---
def main():
    # 1. 初始化环境和智能体
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DDQNAgent(state_dim, action_dim)

    total_steps = 0
    num_episodes = 500

    # 2. 训练循环
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        while True:
            # 3. 智能体选择动作
            action = agent.select_action(state)

            # 4. 环境执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 5. 将经验存入回放池
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # 6. 状态转移
            state = next_state
            episode_reward += reward
            total_steps += 1

            # 7. 智能体学习
            agent.learn()

            # 8. 定期更新目标网络
            if total_steps % TARGET_UPDATE_FREQUENCY == 0:
                agent.update_target_network()

            if done:
                break

        print(f"Episode: {episode + 1}, Total Reward: {episode_reward}")

    env.close()


if __name__ == '__main__':
    main()