import time
from typing import Union

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
import gymnasium as gym


class VisualizationCallback(BaseCallback):
    def __init__(self, eval_env: Union[gym.Env,VecEnv], eval_freq: int, verbose: int = 1):
        super(VisualizationCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self._is_vec_env = isinstance(self.eval_env, VecEnv)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            if self.verbose > 0:
                print(f"\n--- 开始在第 {self.num_timesteps} 步进行可视化 ---")

            if self._is_vec_env:
                obs = self.eval_env.reset()
            else:
                obs, info = self.eval_env.reset()

            for _ in range(1000):
                action, _states = self.model.predict(obs, deterministic=True)

                if self._is_vec_env:
                    obs, reward, dones, infos = self.eval_env.step(action)
                    done = dones[0]
                    self.eval_env.render('human')
                else:
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    done = terminated or truncated

                time.sleep(0.01)  # 稍微加长一点延迟，否则Atari游戏闪太快

                if done:
                    break

            if self.verbose > 0:
                print("--- 可视化结束, 继续训练 ---\n")

        return True