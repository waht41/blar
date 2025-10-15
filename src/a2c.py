import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import gymnasium as gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


# --- 1. 定义 Actor-Critic 网络 ---
# Actor和Critic共享网络的前几层，以提取共同的状态特征表示
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()

        # 共享层
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor-specific layer (策略头)
        # 输出每个动作的"logits"，未经softmax处理的原始分数
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Critic-specific layer (价值头)
        # 输出一个标量，代表当前状态的价值
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        前向传播函数
        Args:
            state (Tensor): 环境状态
        Returns:
            action_logits (Tensor): 动作的logits
            state_value (Tensor): 状态的价值
        """
        # 通过共享层
        features = self.shared_layers(state)

        # 计算动作logits和状态价值
        action_logits = self.actor_head(features)
        state_value = self.critic_head(features)

        return action_logits, state_value


# --- 2. 定义 A2C Agent ---
class A2C_Agent:
    def __init__(self, env):
        # --- 超参数设置 ---
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.gamma = 0.99  # 折扣因子
        self.lr = 0.001  # 学习率
        self.n_steps = 5  # N-step returns 的步数 N
        self.value_loss_coef = 0.5  # 价值损失的系数
        self.entropy_coef = 0.01  # 熵奖励的系数

        # --- 初始化网络和优化器 ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = ActorCritic(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def select_action(self, state):
        """根据当前状态选择动作"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action_logits, _ = self.model(state)

        # 从logits创建动作的概率分布
        dist = Categorical(logits=action_logits)

        # 从分布中采样一个动作
        action = dist.sample()

        # 返回动作及其对数概率
        return action.item(), dist.log_prob(action)

    def compute_loss(self, rewards, log_probs, values, masks, next_value):
        """计算A2C的总损失"""
        # rewards, log_probs, values, masks 都是长度为 n_steps 的列表

        # 1. 计算 N-step returns (Q值)
        # G_t = r_t + gamma * r_{t+1} + ... + gamma^N * V(s_{t+N})
        returns = []
        # next_value 是 V(s_{t+N})
        R = next_value
        # 从后往前计算每个时间步的 return
        for i in reversed(range(len(rewards))):
            # masks[i] == 0 代表 s_{i+1} 是终止状态，所以 V(s_{i+1})=0
            R = rewards[i] + self.gamma * R * masks[i]
            returns.insert(0, R)

        returns = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs)
        values = torch.cat(values)

        # 2. 计算优势函数 Advantage = Returns - V(s)
        advantage = returns - values

        # 3. 计算 Actor (策略) 损失
        # L_actor = -log(pi(a|s)) * Advantage
        actor_loss = -(log_probs * advantage.detach()).mean()

        # 4. 计算 Critic (价值) 损失
        # L_critic = (Returns - V(s))^2
        critic_loss = F.mse_loss(values, returns)

        # 5. 计算熵损失 (鼓励探索)
        # 我们希望最大化熵，所以最小化负熵
        # 重新计算分布来获取熵
        # 注意：这里我们只用了最后一个状态的logits来近似计算熵，
        # 完整的做法是保存所有logits并计算每个时间步的熵。
        # 为简化，这里做了近似，在实践中通常效果也不错。
        dist = Categorical(logits=self.last_logits)
        entropy_loss = dist.entropy().mean()

        # 6. 计算总损失
        total_loss = (actor_loss +
                      self.value_loss_coef * critic_loss -
                      self.entropy_coef * entropy_loss)

        return total_loss

    def train(self, max_episodes=1000):
        """主训练函数"""
        episode_rewards = deque(maxlen=100)
        all_rewards = []

        for episode in range(max_episodes):
            state, _ = self.env.reset()
            episode_reward = 0

            # --- 主循环，直到 episode 结束 ---
            while True:
                # --- 1. 进行 N 步的环境交互，收集数据 ---
                log_probs, values, rewards, masks = [], [], [], []

                for _ in range(self.n_steps):
                    action, log_prob = self.select_action(state)

                    # 保存状态对应的价值
                    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                    action_logits, value = self.model(state_tensor)
                    self.last_logits = action_logits  # 保存logits用于计算熵

                    next_state, reward, done, truncated, _ = self.env.step(action)

                    # 存储数据
                    log_probs.append(log_prob)
                    values.append(value)
                    rewards.append(torch.tensor([reward], dtype=torch.float, device=self.device))
                    # done为True时，mask为0；否则为1
                    masks.append(torch.tensor([1 - done], dtype=torch.float, device=self.device))

                    state = next_state
                    episode_reward += reward

                    if done or truncated:
                        break

                # --- 2. 计算 N-step return 的 bootstrap value ---
                # 如果 episode 在 N 步内结束了，则 next_value 为 0
                # 否则，用 Critic 网络估计 V(s_{t+N})
                if done or truncated:
                    next_value = torch.tensor([0.0], device=self.device)
                else:
                    next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
                    _, next_value = self.model(next_state_tensor)
                    next_value = next_value.detach()

                # --- 3. 计算损失并更新网络 ---
                loss = self.compute_loss(rewards, log_probs, values, masks, next_value)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if done or truncated:
                    break

            # --- 记录和打印日志 ---
            episode_rewards.append(episode_reward)
            all_rewards.append(episode_reward)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards)
                print(f"Episode {episode + 1}/{max_episodes}, Average Reward (last 100): {avg_reward:.2f}")

        self.plot_rewards(all_rewards)

    def plot_rewards(self, rewards):
        """绘制奖励曲线"""
        plt.figure(figsize=(10, 5))
        plt.plot(rewards, label='Episode Reward')
        # 计算并绘制移动平均线
        moving_avg = np.convolve(rewards, np.ones(100) / 100, mode='valid')
        plt.plot(np.arange(99, len(rewards)), moving_avg, label='Moving Average (100 episodes)', color='red')
        plt.title('A2C Training on CartPole-v1')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True)
        plt.show()


# --- 3. 运行训练 ---
if __name__ == "__main__":
    # 创建环境
    env = gym.make('CartPole-v1')

    # 创建Agent并开始训练
    agent = A2C_Agent(env)
    agent.train(max_episodes=1500)

    env.close()