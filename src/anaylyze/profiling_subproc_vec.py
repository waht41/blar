import time
import numpy as np
import multiprocessing as mp
import warnings
from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv


# _stack_obs 和 _profiling_worker 保持不变，这里为了完整性一并提供
def _stack_obs(obs_list: Union[list[VecEnvObs], tuple[VecEnvObs]], space: spaces.Space) -> VecEnvObs:
    assert isinstance(obs_list, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs_list) > 0, "need observations from at least one environment"
    if isinstance(space, spaces.Dict):
        assert isinstance(space.spaces, dict), "Dict space must have ordered subspaces"
        assert isinstance(obs_list[0], dict), "non-dict observation for environment with Dict observation space"
        return {key: np.stack([single_obs[key] for single_obs in obs_list]) for key in space.spaces.keys()}
    elif isinstance(space, spaces.Tuple):
        assert isinstance(obs_list[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple(np.stack([single_obs[i] for single_obs in obs_list]) for i in range(obs_len))
    else:
        return np.stack(obs_list)


def _profiling_worker(remote: mp.connection.Connection, parent_remote: mp.connection.Connection,
                      env_fn_wrapper: CloudpickleWrapper) -> None:
    from stable_baselines3.common.env_util import is_wrapped
    parent_remote.close()
    env = _patch_env(env_fn_wrapper.var())
    reset_info: Optional[dict[str, Any]] = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                step_start_time = time.perf_counter()
                observation, reward, terminated, truncated, info = env.step(data)
                step_duration = time.perf_counter() - step_start_time
                info["profiling"] = {"step_duration": step_duration}
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:
                    info["terminal_observation"] = observation
                    observation, reset_info = env.reset()
                remote.send((observation, reward, done, info, reset_info))
            elif cmd == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                observation, reset_info = env.reset(seed=data[0], **maybe_options)
                remote.send((observation, reset_info))
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = env.get_wrapper_attr(data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(env.get_wrapper_attr(data))
            elif cmd == "has_attr":
                try:
                    env.get_wrapper_attr(data)
                    remote.send(True)
                except AttributeError:
                    remote.send(False)
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))  # type: ignore[func-returns-value]
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break
        except KeyboardInterrupt:
            break


# 关键修改：让 ProfilingSubprocVecEnv 返回数据而不是打印
class ProfilingSubprocVecEnv(SubprocVecEnv):
    """
    这个版本的 SubprocVecEnv 会测量耗时, 但不直接打印,
    而是将耗时数据打包到返回的 infos 字典中。
    """

    def __init__(self, env_fns: list[Callable[[], gym.Env]], start_method: Optional[str] = None):
        # __init__ 方法的目标是启动 _profiling_worker，所以内容和之前一样
        super().__init__(env_fns, start_method)
        # 重写 target，确保使用的是我们的分析 worker
        for i, process in enumerate(self.processes):
            process.terminate()

        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            process = ctx.Process(target=_profiling_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

    def step_wait(self) -> VecEnvStepReturn:
        # 测量 IPC 和 Stacking 耗时
        ipc_start_time = time.perf_counter()
        results = [remote.recv() for remote in self.remotes]
        ipc_duration = time.perf_counter() - ipc_start_time

        obs, rews, dones, infos, self.reset_infos = zip(*results)

        # 提取 worker 的耗时
        worker_step_times = [info["profiling"]["step_duration"] for info in infos]

        stack_start_time = time.perf_counter()
        stacked_obs = _stack_obs(obs, self.observation_space)
        stacked_rews = np.stack(rews)
        stacked_dones = np.stack(dones)
        stack_duration = time.perf_counter() - stack_start_time

        # **核心修改**：将耗时数据打包到第一个环境的 info 字典中
        # 这样 PPO 算法就可以收集到它们
        infos[0]["env_profiling"] = {
            "ipc_duration": ipc_duration,
            "stack_duration": stack_duration,
            "avg_worker_time": np.mean(worker_step_times),
            "max_worker_time": np.max(worker_step_times),
        }

        return stacked_obs, stacked_rews, stacked_dones, infos