import os
import sys
import time
import traceback
from multiprocessing import shared_memory
import multiprocessing as mp


import numpy as np

from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv, VecEnvStepReturn, VecEnvObs, \
    VecEnvIndices
from stable_baselines3.common.vec_env.patch_gym import _patch_env


def _shared_memory_profiling_worker(
        worker_id: int,
        remote: mp.connection.Connection,
        parent_remote: mp.connection.Connection,
        env_fn_wrapper: CloudpickleWrapper,
        shm_name: str,
        obs_shape: tuple,
        obs_dtype: np.dtype,
        num_envs: int,
) -> None:
    """
    Worker function that uses shared memory for observations.
    Includes comprehensive debugging and error handling.
    """
    parent_remote.close()
    print(f"[Worker {worker_id}] Process started. PID: {os.getpid()}", flush=True)

    # Connect to the shared memory block
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    shared_obs_buffer = np.ndarray((num_envs,) + obs_shape, dtype=obs_dtype, buffer=existing_shm.buf)
    print(f"[Worker {worker_id}] Shared memory connected.", flush=True)

    # Create the environment
    env = _patch_env(env_fn_wrapper.var())
    print(f"[Worker {worker_id}] Environment created.", flush=True)

    reset_info: dict[str, Any] = {}

    while True:
        try:
            # remote.recv() 会阻塞直到接收到消息
            cmd, data = remote.recv()
            # print(f"[Worker {worker_id}] Command received: {cmd}", flush=True) # 这行可以取消注释用于调试

            if cmd == "step":
                step_start_time = time.perf_counter()
                observation, reward, terminated, truncated, info = env.step(data)
                step_duration = time.perf_counter() - step_start_time
                
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                
                # 记录传输开始时间
                transmission_start_time = time.perf_counter()
                
                # 直接将观察结果写入共享内存
                shared_obs_buffer[worker_id] = observation

                if done:
                    info["terminal_observation"] = observation
                    observation, reset_info = env.reset()

                # 发送除 observation 之外的所有内容，包含时间戳
                info["profiling"] = {
                    "step_duration": step_duration,
                    "transmission_start_time": transmission_start_time
                }
                remote.send((reward, done, info, reset_info))

            elif cmd == "reset":
                reset_info = {}  # 默认值
                maybe_options = {"options": data[1]} if data and len(data) > 1 and data[1] else {}
                seed = data[0] if data else None
                observation, reset_info = env.reset(seed=seed, **maybe_options)

                # 将观察结果写入共享内存
                shared_obs_buffer[worker_id] = observation

                # 优化：只发送 info 字典回来
                remote.send(reset_info)

            elif cmd == "close":
                env.close()
                existing_shm.close()
                remote.close()
                break

            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))

            elif cmd == "render":
                remote.send(env.render())

            elif cmd == "get_attr":
                attr = env.get_wrapper_attr(data)
                remote.send(attr)

            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))  # type: ignore[func-returns-value]

            elif cmd == "env_method":
                method = env.get_wrapper_attr(data[0])
                remote.send(method(*data[1], **data[2]))

            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))

            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")

        except EOFError:
            # 当主进程的连接关闭时，会触发此异常
            print(f"Worker {worker_id} caught EOFError, exiting.")
            break
        except KeyboardInterrupt:
            # 允许通过 Ctrl+C 中断
            print(f"Worker {worker_id} caught KeyboardInterrupt, exiting.")
            break


import multiprocessing as mp
import warnings
from typing import Any, Callable, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Import the original _worker from SubprocVecEnv for initialization
from stable_baselines3.common.vec_env.subproc_vec_env import _worker


class SharedMemoryProfilingVecEnv(VecEnv):
    """
    A multiprocess vectorized environment that uses shared memory for observations
    to reduce IPC overhead, while retaining performance profiling.

    This implementation is based on stable_baselines3.SubprocVecEnv but is
    rewritten to handle its own process lifecycle and shared memory management,
    avoiding inheritance conflicts.

    :param env_fns: A list of functions that will create the environments.
    :param start_method: The method used to start the subprocesses.
    """

    def __init__(self, env_fns: list[Callable[[], gym.Env]], start_method: Optional[str] = None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        # 1. Start a single temporary worker to get observation and action spaces
        # This is the key to correctly initializing the VecEnv base class
        temp_remote, temp_work_remote = ctx.Pipe()
        args = (temp_work_remote, temp_remote, CloudpickleWrapper(env_fns[0]))
        temp_process = ctx.Process(target=_worker, args=args, daemon=True)
        temp_process.start()
        temp_work_remote.close()

        temp_remote.send(("get_spaces", None))
        observation_space, action_space = temp_remote.recv()
        temp_remote.send(("close", None))
        temp_process.join()

        # 2. Now, initialize the base VecEnv class with the correct spaces
        super().__init__(n_envs, observation_space, action_space)

        # 3. Setup shared memory
        assert isinstance(self.observation_space, spaces.Box), "SharedMemoryVecEnv only supports Box observation space."
        obs_shape = self.observation_space.shape
        obs_dtype = self.observation_space.dtype
        buffer_size = self.num_envs * int(np.prod(obs_shape)) * np.dtype(obs_dtype).itemsize
        shm_name = f"sb3_shm_{time.time()}_{np.random.randint(1e9)}"

        try:
            self.shm = shared_memory.SharedMemory(name=shm_name, create=True, size=buffer_size)
        except FileExistsError:
            # This should not happen with a unique name, but as a safeguard:
            shared_memory.SharedMemory(name=shm_name).unlink()
            self.shm = shared_memory.SharedMemory(name=shm_name, create=True, size=buffer_size)

        self.shared_obs_buffer = np.ndarray((self.num_envs,) + obs_shape, dtype=obs_dtype, buffer=self.shm.buf)

        # 4. Start the actual workers using our custom worker function
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for i, (work_remote, remote, env_fn) in enumerate(zip(self.work_remotes, self.remotes, env_fns)):
            args = (i, work_remote, remote, CloudpickleWrapper(env_fn), self.shm.name, obs_shape, obs_dtype,
                    self.num_envs)
            process = ctx.Process(target=_shared_memory_profiling_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        # 记录主进程开始接收数据的时间
        main_process_receive_start = time.perf_counter()
        
        # Receive rewards, dones, and infos from the workers
        remotes_to_monitor = list(self.remotes)
        results = [None] * len(self.remotes)  # 用一个列表按顺序存储结果
        # 如果需要知道是哪个 env 返回的结果，可以创建一个从 remote 到 index 的映射
        remote_to_idx = {remote: i for i, remote in enumerate(self.remotes)}

        # 用于存储每个worker的传输时间
        worker_transmission_times = [0.0] * len(self.remotes)
        
        while len(remotes_to_monitor) > 0:
            # wait会阻塞，直到至少有一个remote准备好被读取
            # 它返回所有就绪的remotes
            ready_remotes = mp.connection.wait(remotes_to_monitor)

            for remote in ready_remotes:
                # 接收结果
                result = remote.recv()
                # 记录接收结束时间
                receive_end_time = time.perf_counter()
                
                # 找到这个remote对应的原始索引
                idx = remote_to_idx[remote]
                # 将结果存入对应的位置
                results[idx] = result
                
                # 计算这个worker的传输时间
                # 从worker的传输开始时间到主进程接收结束时间
                worker_transmission_start = result[2].get("profiling", {}).get("transmission_start_time", main_process_receive_start)
                worker_transmission_times[idx] = receive_end_time - worker_transmission_start
                
                # 将这个remote从监控列表中移除，因为它已经处理完了
                remotes_to_monitor.remove(remote)
        main_process_receive_end = time.perf_counter()
        
        self.waiting = False
        rews, dones, infos, self.reset_infos = zip(*results)

        # Get observations directly from shared memory
        stack_start_time = time.perf_counter()
        obs = self.shared_obs_buffer.copy()
        stack_duration = time.perf_counter() - stack_start_time

        # 计算平均IPC传输时间
        actual_ipc_duration = np.mean(worker_transmission_times)
        
        # 计算主进程接收数据的时间（不包含worker传输时间）
        main_process_receive_duration = main_process_receive_end - main_process_receive_start

        # Aggregate profiling data
        worker_step_times = [info.get("profiling", {}).get("step_duration", 0) for info in infos]
        if "env_profiling" not in infos[0]:
            infos[0]["env_profiling"] = {}
        infos[0]["env_profiling"].update({
            "ipc_duration": actual_ipc_duration,  # 平均IPC传输时间
            "worker_transmission_times": worker_transmission_times,  # 各个worker的传输时间
            "max_worker_transmission_time": np.max(worker_transmission_times),  # 最大传输时间
            "min_worker_transmission_time": np.min(worker_transmission_times),  # 最小传输时间
            "main_process_receive_duration": main_process_receive_duration,  # 主进程接收时间
            "stack_duration": stack_duration,
            "avg_worker_time": np.mean(worker_step_times),
            "max_worker_time": np.max(worker_step_times),
        })

        return obs, np.stack(rews), np.stack(dones), list(infos)

    def reset(self) -> VecEnvObs:
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", (self._seeds[env_idx], self._options[env_idx])))

        # Receive only the reset_infos back
        self.reset_infos = [remote.recv() for remote in self.remotes]

        # Get the new observations from shared memory
        obs = self.shared_obs_buffer.copy()

        self._reset_seeds()
        self._reset_options()
        return obs

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()

        # Clean up shared memory
        try:
            self.shm.close()
            self.shm.unlink()
        except FileNotFoundError:
            pass  # Already unlinked

        self.closed = True

    # ========================================================================
    # Boilerplate methods copied from SubprocVecEnv to fulfill the interface
    # ========================================================================

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()  # Wait for acknowledgement

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None) -> list[bool]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_class))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices: VecEnvIndices) -> list[Any]:
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]