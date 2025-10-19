import time

import numpy as np
import torch
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor  # 导入 obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv


class CustomPPO(PPO):
    """
    一个自定义的 PPO 类，它重写了 collect_rollouts 方法，
    以便对数据收集过程中的各个关键阶段进行精确的性能分析。
    此版本与较新的 SB3 版本兼容，包含了对环境超时的处理。
    """

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
            n_rollout_steps: int,
    ) -> bool:
        """
        收集经验并存入 buffer，并在最后打印详细的性能分析总结。
        """
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        # ==================== 计时器和数据收集列表初始化 ====================
        total_predict_time = 0.0
        total_env_step_time = 0.0
        total_timeout_handling_time = 0.0
        total_buffer_add_time = 0.0

        # 新增：用于收集每个 step 的环境内部耗时
        env_profiling_data = []

        rollout_start_time = time.perf_counter()
        # =================================================================

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)

                # --- 1. 测量模型推断时间 ---
                start_time = time.perf_counter()
                actions, values, log_probs = self.policy(obs_tensor)
                total_predict_time += time.perf_counter() - start_time
                # -------------------------

            actions = actions.cpu().numpy()
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            # --- 2. 测量环境交互时间 ---
            start_time = time.perf_counter()
            new_obs, rewards, dones, infos = env.step(clipped_actions)
            total_env_step_time += time.perf_counter() - start_time
            # ---------------------------

            # **从 infos 中收集环境分析数据**
            if "env_profiling" in infos[0]:
                env_profiling_data.append(infos[0]["env_profiling"])

            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            # --- 3. 测量环境超时处理时间 ---
            start_time = time.perf_counter()
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value
            total_timeout_handling_time += time.perf_counter() - start_time
            # ---------------------------------

            # --- 4. 测量数据添加到 Buffer 的时间 ---
            start_time = time.perf_counter()
            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            total_buffer_add_time += time.perf_counter() - start_time
            # ----------------------------------------

            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()

        # ==================== 打印最终的性能总结报告 🎯 ====================
        total_rollout_time = time.perf_counter() - rollout_start_time
        other_time = total_rollout_time - total_predict_time - total_env_step_time - total_timeout_handling_time - total_buffer_add_time

        print("\n" + "=" * 60)
        print("Rollout Performance Summary (per rollout):")
        print(f"  - Total Rollout Time:              {total_rollout_time:.4f}s")
        print(
            f"    - Model Inference (Total):       {total_predict_time:.4f}s ({total_predict_time / total_rollout_time:.1%})")
        print(
            f"    - Buffer Add (Total):            {total_buffer_add_time:.4f}s ({total_buffer_add_time / total_rollout_time:.1%})")

        # 打印环境相关的平均统计数据
        if env_profiling_data:
            # 计算各项指标的平均值
            avg_ipc = np.mean([d["ipc_duration"] for d in env_profiling_data])
            avg_main_receive = np.mean([d.get("main_process_receive_duration", 0) for d in env_profiling_data])
            avg_stack = np.mean([d["stack_duration"] for d in env_profiling_data])
            avg_worker = np.mean([d["avg_worker_time"] for d in env_profiling_data])
            total_avg_step_wait = avg_ipc + avg_stack

            print(
                f"  - Environment Interaction (Total): {total_env_step_time:.4f}s ({total_env_step_time / total_rollout_time:.1%})")
            print(f"    └── Breakdown (Avg. per step):")
            print(f"        ├── Total Main Process Wait:   {total_avg_step_wait:.4f}s")
            print(f"        │   ├── Actual IPC Transfer:   {avg_ipc:.4f}s ({avg_ipc / total_avg_step_wait:.1%})")
            print(f"        │   │   ├── Worker→Main:       {avg_ipc:.4f}s")
            print(f"        │   │   └── Main Receive Only: {avg_main_receive:.4f}s")
            print(f"        │   └── Data Stacking:         {avg_stack:.4f}s ({avg_stack / total_avg_step_wait:.1%})")
            print(f"        └── Avg. Worker env.step():    {avg_worker:.4f}s")
        else:
            # 如果没有收到分析数据，则只打印总的环境耗时
            print(
                f"  - Environment Interaction (Total): {total_env_step_time:.4f}s ({total_env_step_time / total_rollout_time:.1%})")

        print(
            f"    - Timeout Handling (Total):      {total_timeout_handling_time:.4f}s ({total_timeout_handling_time / total_rollout_time:.1%})")
        print(f"    - Other Overhead (Total):        {other_time:.4f}s ({other_time / total_rollout_time:.1%})")
        print("=" * 60 + "\n")

        return True