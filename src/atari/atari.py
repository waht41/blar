import ale_py # ale_py 需要被 import 一次，让 gym 能够发现 Atari 环境
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CallbackList
import os
from typing import Callable  # 导入 Callable 用于定义学习率调度

from src.callbacks import VisualizationCallback, PerformanceCallbackWithTqdm
from src.utils.training_utils import setup_training_args_and_logs, print_training_header, print_training_footer
ale_py # 引入环境，用来让import不被ide自动删除

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    创建一个线性学习率衰减的调度器。
    :param initial_value: 初始学习率
    :return: 一个函数，输入进度(0-1)，输出当前学习率
    """

    def func(progress_remaining: float) -> float:
        """
        随着训练的进行，学习率从 initial_value 线性降低到 0.
        progress_remaining 从 1 线性降低到 0.
        """
        return progress_remaining * initial_value

    return func


def main():
    """主函数：训练Atari Breakout PPO模型"""
    # --- 1. 解析命令行参数和设置日志目录 ---
    args, log_dir, model_path = setup_training_args_and_logs(
        game_name="AtariBreakout",
        model_name="atari_breakout_model"
    )

    # gym.register_envs(ale_py) # 这行是不需要的，只要 ale_py 被导入，gym 就会自动注册环境

    print_training_header("Atari Breakout")

    # --- 2. 创建环境 ---
    print("📦 正在创建训练环境...")
    # 使用 make_atari_env 会自动应用一系列关键的 Wrapper，例如帧跳过(Frame Skipping)
    n_envs = 16
    train_env = make_atari_env('ALE/Breakout-v5', n_envs=n_envs, seed=0)
    # VecFrameStack 将连续的4帧图像堆叠起来，让智能体能感知到运动方向
    train_env = VecFrameStack(train_env, n_stack=4)
    print(f"✅ 训练环境创建完成，使用 {n_envs} 个并行环境")

    eval_env = make_atari_env('ALE/Breakout-v5', n_envs=1, seed=1)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    print("✅ 可视化环境创建完成")

    # --- 3. 实例化回调函数 ---
    print("🎯 设置回调函数...")
    # 由于总步数增加，可以适当增加回调频率
    vis_callback = VisualizationCallback(eval_env, eval_freq=50000)  # eval_freq 是基于单个环境的步数
    performance_callback = PerformanceCallbackWithTqdm(verbose=1)
    
    # 组合多个回调函数
    callbacks = CallbackList([vis_callback, performance_callback])
    print("✅ 回调函数设置完成 (可视化 + 性能监控)")

    # --- 4. 定义超参数并创建或加载 PPO 模型 ---

    # 关键优化：为Atari设置一套更稳定的超参数
    # 参考了 Stable Baselines3 Zoo 和相关论文的推荐值
    learning_rate = 2.5e-4
    ppo_params = {
        'n_steps': 2048,  # 增加每次更新收集的样本数，以获得更稳定的梯度估计
        'batch_size': 512,  # 增加 mini-batch 的大小
        'n_epochs': 4,  # 减少 epoch 数量，防止在当前数据上过拟合
        'gamma': 0.99,  # 折扣因子
        'gae_lambda': 0.95,  # GAE-Lambda 参数
        'clip_range': 0.1,  # 减小 clip_range，限制策略更新幅度，提升稳定性
        'ent_coef': 0.01,  # 熵系数，鼓励探索
        'vf_coef': 0.5,  # 价值函数系数
        'learning_rate': linear_schedule(learning_rate),  # 使用线性衰减的学习率，从 2.5e-4 降到 0
    }

    if args.resume:
        print("🔄 正在加载现有模型...")
        if not os.path.exists(args.resume):
            print(f"❌ 错误：模型文件不存在: {args.resume}")
            return

        model = PPO.load(
            args.resume,
            env=train_env,
            reset_num_timesteps=False,
            # 加载模型时也最好传入自定义参数，以防默认参数覆盖
            custom_objects={"learning_rate": ppo_params['learning_rate']}
        )
        print(f"✅ 模型加载成功: {args.resume}")
        print("📊 将继续之前的训练进度")
    else:
        print("🆕 正在创建新的PPO模型...")
        model = PPO(
            "CnnPolicy",
            train_env,
            verbose=1,
            tensorboard_log=log_dir,
            device='auto',
            **ppo_params  # 使用 ** 解包字典，传入所有优化后的超参数
        )
        print("✅ 新PPO模型创建完成")
        print("--- 使用的超参数 ---")
        for key, value in ppo_params.items():
            # 对 learning_rate 做特殊处理，因为它是一个函数
            if key == 'learning_rate':
                print(f"{key}: linear_schedule({learning_rate})")
            else:
                print(f"{key}: {value}")
        print("--------------------")

    print("\n" + "=" * 60)
    print("🏃‍♂️ 开始训练...")
    print("=" * 60)

    # 关键优化：增加总训练步数。Atari游戏需要大量样本来学习
    # 100万步对于Atari来说仅仅是开始，通常需要1000万步或更多
    TOTAL_TIMESTEPS = 3_000_000

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name="PPO_with_vis",
        callback=callbacks
    )

    print_training_footer(log_dir)

    # --- 5. 保存最终模型 ---
    print("💾 正在保存最终模型...")
    model.save(model_path)
    print(f"✅ 最终模型已保存至: {model_path}")

    # --- 6. 清理环境 ---
    print("🧹 正在清理环境...")
    train_env.close()
    eval_env.close()
    print("✅ 环境清理完成")
    print("👋 程序结束")


if __name__ == "__main__":
    main()