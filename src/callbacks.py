import os
import time
from datetime import datetime
from typing import Union

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from tqdm.auto import trange


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


class PerformanceCallbackWithTqdm(BaseCallback):
    """
    一个使用tqdm进度条来实时监控训练性能指标的Callback。

    进度条会显示:
    - FPS: 每秒处理的Timestep数量 (Frames Per Second)，由tqdm平滑计算
    - sample_time: 最近几次采集Rollout样本的平均时间（秒）
    - update_time: 最近几次模型更新的平均时间（秒）
    - remaining_h: 预估的剩余训练时间（小时）

    :param verbose: (int) 日志级别
    """

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        # --- 内部计时器 ---
        self.rollout_start_time = 0
        self.model_update_start_time = 0

        # --- 指标存储 ---
        # 使用列表来存储最近几次的时间，取平均值可以让显示更平滑
        self.recent_sample_times = []
        self.recent_update_times = []
        self.latest_fps = 0

        # tqdm 进度条实例
        self.pbar = None

    def _on_training_start(self) -> None:
        """在训练开始时被调用，初始化tqdm进度条"""
        # 创建tqdm进度条，总步数为 model.learn() 中设置的目标
        # `leave=True` 确保训练结束后进度条会保留在屏幕上
        self.pbar = trange(
            self.model._total_timesteps,
            desc="Training Progress",
            unit="timestep",
            leave=True
        )
        if self.verbose > 0:
            print("--- TQDM 性能监控已启动 ---")

    def _on_rollout_start(self) -> None:
        """在每一次新的Rollout开始时被调用，计算模型更新时间"""
        # 如果不是第一次Rollout，就计算上一次的模型更新耗时
        if self.model_update_start_time > 0:
            model_update_time = time.time() - self.model_update_start_time
            self.logger.record("custom/time/model_update_s", model_update_time)

            # 存储最近的更新时间
            self.recent_update_times.append(model_update_time)
            # 只保留最近5次的值，防止列表无限增长并用于计算移动平均
            if len(self.recent_update_times) > 5:
                self.recent_update_times.pop(0)

        self.rollout_start_time = time.time()

    def _on_rollout_end(self) -> None:
        """在每一次Rollout结束时被调用，计算样本采集时间"""
        sample_collection_time = time.time() - self.rollout_start_time
        self.logger.record("custom/time/sample_collection_s", sample_collection_time)

        self.recent_sample_times.append(sample_collection_time)
        if len(self.recent_sample_times) > 5:
            self.recent_sample_times.pop(0)

        # 标记模型更新即将开始
        self.model_update_start_time = time.time()

    def _on_step(self) -> bool:
        """在环境执行每一步后被调用，更新进度条"""
        # 1. 更新进度条的当前步数
        # self.pbar.n 是进度条的当前计数值, self.num_timesteps 是模型已训练的总步数
        # 这个差值就是自上次更新以来新走的步数
        self.pbar.update(self.num_timesteps - self.pbar.n)

        # 2. 从tqdm的内部统计数据中获取平滑后的FPS
        # tqdm的 `format_dict` 包含了很多有用的信息，包括平滑后的速率 (rate)
        # 这个速率就是 timesteps/second (FPS)，比我们自己算要更稳定
        tqdm_stats = self.pbar.format_dict
        if 'rate' in tqdm_stats and tqdm_stats['rate'] is not None:
            self.latest_fps = tqdm_stats['rate']
            self.logger.record("custom/fps", self.latest_fps)

        # 3. 准备要显示在进度条右侧的字典信息
        postfix_dict = {}
        if self.latest_fps > 0:
            postfix_dict["fps"] = f"{self.latest_fps:.1f}"

            # 根据当前实时FPS估算剩余时间
            remaining_timesteps = self.model._total_timesteps - self.num_timesteps
            remaining_time_s = remaining_timesteps / self.latest_fps
            postfix_dict["remaining_h"] = f"{remaining_time_s / 3600:.2f}"

        if self.recent_sample_times:
            # 显示最近几次采集时间的平均值，结果更平滑
            postfix_dict["sample_time"] = f"{np.mean(self.recent_sample_times):.3f}s"

        if self.recent_update_times:
            # 显示最近几次更新时间的平均值
            postfix_dict["update_time"] = f"{np.mean(self.recent_update_times):.3f}s"

        # 4. 设置进度条的后缀信息
        if postfix_dict:
            self.pbar.set_postfix(postfix_dict, refresh=False)  # refresh=False 由tqdm自动管理刷新

        return True

    def _on_training_end(self) -> None:
        """在训练结束时调用，关闭进度条"""
        if self.pbar:
            self.pbar.close()
            self.pbar = None
        if self.verbose > 0:
            print("--- TQDM 性能监控已结束 ---")


class ModelSaveCallback(BaseCallback):
    """
    定期保存模型的回调函数。
    
    每隔指定的步数自动保存模型，防止训练中断导致模型丢失。
    
    :param save_freq: (int) 保存频率，每隔多少步保存一次模型
    :param save_path: (str) 模型保存路径
    :param name_prefix: (str) 模型文件名前缀
    :param verbose: (int) 日志级别
    """
    
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "model", verbose: int = 1):
        super(ModelSaveCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        
        # 确保保存目录存在
        os.makedirs(self.save_path, exist_ok=True)
        
        if self.verbose > 0:
            print(f"✅ 模型保存回调已初始化")
            print(f"   - 保存频率: 每 {self.save_freq:,} 步")
            print(f"   - 保存路径: {self.save_path}")
            print(f"   - 文件名前缀: {self.name_prefix}")
    
    def _on_step(self) -> bool:
        """在每一步后检查是否需要保存模型"""
        if self.num_timesteps % self.save_freq == 0 and self.num_timesteps > 0:
            # 生成带时间戳的模型文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{self.name_prefix}_step_{self.num_timesteps}_{timestamp}.zip"
            model_path = os.path.join(self.save_path, model_filename)
            
            # 保存模型
            self.model.save(model_path)
            
            if self.verbose > 0:
                print(f"💾 模型已保存: {model_path}")
                print(f"   - 当前步数: {self.num_timesteps:,}")
                print(f"   - 下次保存: 第 {self.num_timesteps + self.save_freq:,} 步")
        
        return True
