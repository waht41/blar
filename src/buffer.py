import numpy as np
import torch as th
from gymnasium import spaces
from typing import Generator, Optional, Union, NamedTuple

from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.vec_env import VecNormalize


# 确保 RolloutBufferSamples 类型定义是可用的
# 如果你使用的是较新版本的 SB3，它可能已经是一个 dataclass
class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class RolloutGPUBuffer(BaseBuffer):
    """
    Rollout buffer optimized for GPU, used in on-policy algorithms like PPO.
    It stores all data as PyTorch Tensors directly on the GPU.

    To maintain compatibility with the original PPO's logging which expects NumPy arrays,
    it uses @property decorators for `.values` and `.returns`. These properties
    perform a .cpu().numpy() conversion only when explicitly accessed from outside,
    thus not affecting the performance of the training loop.
    """

    # 内部存储始终使用 PyTorch Tensors
    observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    advantages: th.Tensor
    episode_starts: th.Tensor
    log_probs: th.Tensor
    _values: th.Tensor
    _returns: th.Tensor

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            device: Union[th.device, str] = "auto",
            gae_lambda: float = 1,
            gamma: float = 0.99,
            n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    # 改变1: 创建 @property，用于外部代码访问
    @property
    def values(self) -> np.ndarray:
        """
        Returns the V-values as a NumPy array on the CPU.
        Accessed by PPO for logging explained_variance. Conversion happens on-demand.
        """
        return self._values.flatten().cpu().numpy()

    @property
    def returns(self) -> np.ndarray:
        """
        Returns the returns (TD(lambda) estimates) as a NumPy array on the CPU.
        """
        return self._returns.flatten().cpu().numpy()

    def reset(self) -> None:
        obs_shape = self.obs_shape
        # Handle Discrete observation space
        if isinstance(self.observation_space, spaces.Discrete):
            obs_shape = (1,)

        # 改变2: 初始化内部私有变量 (带下划线)
        self.observations = th.zeros((self.buffer_size, self.n_envs, *obs_shape), dtype=th.float32, device=self.device)
        self.actions = th.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=th.float32, device=self.device)
        self.rewards = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32, device=self.device)
        self.episode_starts = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32, device=self.device)
        self.log_probs = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32, device=self.device)
        self.advantages = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32, device=self.device)
        self._values = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32, device=self.device)
        self._returns = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32, device=self.device)

        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        last_values = last_values.flatten()
        dones_tensor = th.as_tensor(dones.astype(np.float32), device=self.device)

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones_tensor
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                # 改变3: 内部读取私有变量
                next_values = self._values[step + 1]

            # 改变4: 内部读取私有变量
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self._values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        # 改变5: 内部写入私有变量
        self._returns = self.advantages + self._values

    def add(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            episode_start: np.ndarray,
            value: th.Tensor,
            log_prob: th.Tensor,
    ) -> None:
        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)

        # Reshape to handle multi-dim and discrete action spaces
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to GPU
        _obs = th.as_tensor(obs, device=self.device)
        _action = th.as_tensor(action, device=self.device)
        _reward = th.as_tensor(reward, device=self.device)
        _episode_start = th.as_tensor(episode_start, device=self.device)

        self.observations[self.pos] = _obs
        self.actions[self.pos] = _action
        self.rewards[self.pos] = _reward
        self.episode_starts[self.pos] = _episode_start
        # 改变6: 内部写入私有变量
        self._values[self.pos] = value.flatten()
        self.log_probs[self.pos] = log_prob
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        if not self.generator_ready:
            total_samples = self.buffer_size * self.n_envs

            self.observations = self.observations.reshape(total_samples, *self.obs_shape)
            self.actions = self.actions.reshape(total_samples, self.action_dim)
            self.log_probs = self.log_probs.reshape(total_samples)
            self.advantages = self.advantages.reshape(total_samples)
            # 改变7: 内部 reshape 私有变量
            self._values = self._values.reshape(total_samples)
            self._returns = self._returns.reshape(total_samples)

            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
            self,
            batch_inds: np.ndarray,
            env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples:
        # 改变8: 在返回的样本中使用私有变量 (它们是Tensor)
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self._values[batch_inds],
            self.log_probs[batch_inds],
            self.advantages[batch_inds],
            self._returns[batch_inds],
        )
        # The named tuple for PPO expects `old_values` and `old_log_prob`
        return RolloutBufferSamples(*data)