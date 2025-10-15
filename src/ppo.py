import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. 经验缓冲区 ---
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.values[:]


# --- 2. Actor-Critic 网络 ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


# --- 3. PPO Agent ---
class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, gae_lambda):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda  # 新增 GAE lambda 参数

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, logprob, state_val = self.policy_old.act(state)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(logprob)
        memory.values.append(state_val)

        return action.item()

    def update(self, memory):
        # --- 转换数据为 tensor ---
        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(device)
        old_values = torch.squeeze(torch.stack(memory.values, dim=0)).detach().to(device)

        # --- 使用 GAE 计算优势 ---
        advantages = []
        last_gae_lam = 0
        # 从后向前遍历
        for i in reversed(range(len(memory.rewards))):
            if i == len(memory.rewards) - 1:
                next_non_terminal = 0  # 最后一个状态没有下一个状态
                next_value = 0  # 最后一个状态没有下一个价值
            else:
                next_non_terminal = 1.0 - memory.dones[i]
                next_value = old_values[i + 1]

            # 计算 TD 误差
            delta = memory.rewards[i] + self.gamma * next_value * next_non_terminal - old_values[i]
            # GAE 公式
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages.insert(0, last_gae_lam)

        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        # 价值函数的训练目标是优势加上旧的价值估计
        value_targets = advantages + old_values

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # 用于记录损失的列表
        actor_losses, critic_losses, entropy_losses = [], [], []

        # --- 训练 K 个 Epochs ---
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # 损失计算
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.MseLoss(state_values, value_targets.unsqueeze(1))
            entropy_loss = dist_entropy.mean()

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 记录损失值
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropy_losses.append(entropy_loss.item())

        self.policy_old.load_state_dict(self.policy.state_dict())

        # 返回平均损失
        return np.mean(actor_losses), np.mean(critic_losses), np.mean(entropy_losses)


# --- 4. 主训练循环 ---
if __name__ == '__main__':
    ############## 超参数 ##############
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    render = False
    log_interval = 10  # 减少间隔以更频繁地查看日志
    max_episodes = 500
    max_timesteps = 400

    update_timestep = 2000

    K_epochs = 40  # 可以适当减少，因为GAE更稳定
    eps_clip = 0.2
    gamma = 0.99
    gae_lambda = 0.95  # GAE 的 lambda 参数

    lr_actor = 0.0003
    lr_critic = 0.001

    random_seed = 0
    ####################################

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, gae_lambda)

    # 记录日志
    running_reward = 0
    avg_length = 0
    time_step = 0
    total_actor_loss, total_critic_loss, total_entropy_loss = 0, 0, 0
    update_count = 0

    # 训练循环
    for i_episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        for t in range(max_timesteps):
            time_step += 1

            action = ppo.select_action(state, memory)
            state, reward, done, truncated, _ = env.step(action)

            memory.rewards.append(reward)
            memory.dones.append(done or truncated)
            episode_reward += reward

            if time_step % update_timestep == 0:
                print(f"--- Episode {i_episode}: Performing update ---")  # 增加更新提示
                actor_loss, critic_loss, entropy_loss = ppo.update(memory)
                memory.clear_memory()
                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
                total_entropy_loss += entropy_loss
                update_count += 1
                time_step = 0

            if render:
                env.render()

            if done or truncated:
                break

        running_reward += episode_reward
        avg_length += t

        # 打印日志
        if i_episode % log_interval == 0:
            avg_length = avg_length / log_interval
            running_reward = running_reward / log_interval

            avg_actor_loss = total_actor_loss / update_count if update_count > 0 else 0
            avg_critic_loss = total_critic_loss / update_count if update_count > 0 else 0
            avg_entropy = total_entropy_loss / update_count if update_count > 0 else 0

            print(f"Episode {i_episode:<5} | "
                  f"Avg Length: {avg_length:<5.1f} | "
                  f"Avg Reward: {running_reward:<6.2f}")
            print(f"           | "
                  f"Actor Loss: {avg_actor_loss:<7.4f} | "
                  f"Critic Loss: {avg_critic_loss:<7.4f} | "
                  f"Entropy: {avg_entropy:<7.4f}")
            print("-" * 70)

            # 重置记录器
            running_reward = 0
            avg_length = 0
            total_actor_loss, total_critic_loss, total_entropy_loss = 0, 0, 0
            update_count = 0

    env.close()