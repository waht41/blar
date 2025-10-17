import time
from typing import Union
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
import gymnasium as gym

class VisualizationCallback(BaseCallback):
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        eval_freq: int,
        fps: int = 30,
        max_duration_seconds: float = 30.0,
        verbose: int = 1
    ):
        super(VisualizationCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.fps = fps
        self.max_duration_seconds = max_duration_seconds
        self._is_vec_env = isinstance(self.eval_env, VecEnv)
        self.sleep_time = 1.0 / self.fps  # 计算每帧之间的延迟时间

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            if self.verbose > 0:
                print(f"\n--- 开始在第 {self.num_timesteps} 步进行可视化 (FPS: {self.fps}, 最长展示: {self.max_duration_seconds}s) ---")

            if self._is_vec_env:
                obs = self.eval_env.reset()
            else:
                obs, info = self.eval_env.reset()

            start_time = time.time()  # 记录可视化开始时间
            step_count = 0
            max_steps = int(self.max_duration_seconds * self.fps)  # 计算最大步数

            for _ in range(max_steps):
                step_start = time.time()  # 记录单步开始时间
                action, _states = self.model.predict(obs, deterministic=True)

                if self._is_vec_env:
                    obs, reward, dones, infos = self.eval_env.step(action)
                    done = dones[0]
                    self.eval_env.render('human')
                else:
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    done = terminated or truncated

                # 计算本帧剩余需要睡眠的时间
                elapsed = time.time() - step_start
                sleep_duration = max(0.0, self.sleep_time - elapsed)
                time.sleep(sleep_duration)

                step_count += 1
                if done:
                    break

            total_time = time.time() - start_time
            if self.verbose > 0:
                status = "环境提前结束" if step_count < max_steps else "达到最大展示时间"
                print(f"--- 可视化结束 ({status}, 时长: {total_time:.1f}s, 步数: {step_count}) ---\n")

        return True