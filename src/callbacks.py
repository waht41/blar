import time

from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym

class VisualizationCallback(BaseCallback):
    """
    一个自定义的回调函数，用于在训练过程中周期性地可视化智能体的表现。

    :param eval_env: 用于评估和渲染的环境。
    :param eval_freq: 每隔多少个 'timesteps' 进行一次可视化。
    """

    def __init__(self, eval_env: gym.Env, eval_freq: int, verbose: int = 1):
        super(VisualizationCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        """
        这个方法会在训练的每一步之后被调用。
        我们检查是否达到了可视化的频率。
        """
        # self.num_timesteps 是 BaseCallback 中记录的全局时间步数
        if self.num_timesteps % self.eval_freq == 0:
            if self.verbose > 0:
                print(f"\n--- 开始在第 {self.num_timesteps} 步进行可视化 ---")

            obs, info = self.eval_env.reset()
            # 运行最多 1000 帧或直到任务结束
            for _ in range(1000):
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)

                # 短暂延迟，让人眼能看清
                time.sleep(0.01)

                if terminated or truncated:
                    break

            if self.verbose > 0:
                print("--- 可视化结束, 继续训练 ---\n")

        return True  # 返回 True 表示继续训练