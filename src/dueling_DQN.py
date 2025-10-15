import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import random
from collections import deque, namedtuple

# --- 超参数定义 ---
BUFFER_SIZE = int(1e5)  # Replay buffer 的大小
BATCH_SIZE = 64  # 每次训练的 mini-batch 大小
GAMMA = 0.99  # 折扣因子
TAU = 1e-3  # 用于目标网络的软更新
LR = 5e-4  # 学习率
UPDATE_EVERY = 4  # 每隔多少步更新一次网络

# --- 设备配置 (GPU or CPU) ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Dueling Q-Network 定义 ---
class DuelingQNetwork(nn.Module):
    """实现了 Dueling DQN 架构的神经网络"""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """
        初始化模型.
        参数:
            state_size (int): 状态空间的维度
            action_size (int): 动作空间的维度
            seed (int): 随机种子
            fc1_units (int): 第一个隐藏层的节点数
            fc2_units (int): 第二个隐藏层的节点数
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # 共享的网络层
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        # 状态价值流 (Value Stream)
        self.value_stream = nn.Linear(fc2_units, 1)

        # 优势函数流 (Advantage Stream)
        self.advantage_stream = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """
        前向传播，将状态映射到动作价值.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # 计算 V(s)
        V = self.value_stream(x)

        # 计算 A(s, a)
        A = self.advantage_stream(x)

        # 聚合 V(s) 和 A(s, a) 得到 Q(s, a)
        # 公式: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
        # A.mean(1, keepdim=True) 计算每个状态下所有动作优势的平均值
        Q = V + (A - A.mean(1, keepdim=True))

        return Q


# --- Replay Buffer 定义 ---
class ReplayBuffer:
    """固定大小的缓冲区，用于存储经验元组"""

    def __init__(self, buffer_size, batch_size, seed):
        """
        初始化 ReplayBuffer.
        参数:
            buffer_size (int): 缓冲区最大大小
            batch_size (int): 每个训练批次的大小
            seed (int): 随机种子
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """向 memory 中添加新的经验"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """从 memory 中随机采样一个批次的经验"""
        experiences = random.sample(self.memory, k=self.batch_size)

        # 将经验元组转换为 Torch Tensors
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """返回当前 memory 中的经验数量"""
        return len(self.memory)


# --- Agent 定义 ---
class Agent():
    """与环境交互并学习的 Agent"""

    def __init__(self, state_size, action_size, seed):
        """
        初始化 Agent.
        参数:
            state_size (int): 状态空间维度
            action_size (int): 动作空间维度
            seed (int): 随机种子
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_policy = DuelingQNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(device)
        # 初始化时，让 target 网络和 policy 网络的权重相同
        self.qnetwork_target.load_state_dict(self.qnetwork_policy.state_dict())

        self.optimizer = optim.Adam(self.qnetwork_policy.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # 保存经验到 replay buffer
        self.memory.add(state, action, reward, next_state, done)

        # 每隔 UPDATE_EVERY 步，更新一次网络
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def select_action(self, state, eps=0.):
        """
        根据 epsilon-greedy 策略选择动作.
        参数:
            state (array_like): 当前状态
            eps (float): epsilon 值，用于 epsilon-greedy
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_policy.eval()
        with torch.no_grad():
            action_values = self.qnetwork_policy(state)
        self.qnetwork_policy.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """
        使用给定的一个批次的经验元组来更新价值函数.
        参数:
            experiences (Tuple[torch.Tensor]): (s, a, r, s', done) 元组
            gamma (float): 折扣因子
        """
        states, actions, rewards, next_states, dones = experiences

        # --- DDQN 核心部分 ---
        # 1. 使用 policy network 选择下一个状态的最佳动作
        best_actions_next = self.qnetwork_policy(next_states).detach().max(1)[1].unsqueeze(1)
        # 2. 使用 target network 评估这些动作的 Q 值
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, best_actions_next)

        # 计算目标 Q 值
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # 从 policy network 中获取期望的 Q 值
        Q_expected = self.qnetwork_policy(states).gather(1, actions)

        # 计算损失
        loss = F.mse_loss(Q_expected, Q_targets)

        # 最小化损失
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # --- 更新目标网络 ---
        self.soft_update(self.qnetwork_policy, self.qnetwork_target, TAU)

    def soft_update(self, policy_model, target_model, tau):
        """
        软更新模型参数.
        θ_target = τ*θ_policy + (1 - τ)*θ_target
        """
        for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)


# --- 训练函数 ---
def train():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)

    n_episodes = 2000
    max_t = 1000
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 0.995

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start

    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.select_action(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
        if np.mean(scores_window) >= 195.0:  # CartPole-v1 的解决标准
            print(f'\nEnvironment solved in {i_episode - 100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            # 保存模型权重
            torch.save(agent.qnetwork_policy.state_dict(), 'dueling_dqn_checkpoint.pth')
            break

    env.close()


if __name__ == '__main__':
    train()