import gymnasium as gym
import ale_py


from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import os

from src.callbacks import VisualizationCallback
from src.utils.training_utils import setup_training_args_and_logs, print_training_header, print_training_footer


def main():
    """主函数：训练Atari Breakout PPO模型"""
    # --- 1. 解析命令行参数和设置日志目录 ---
    args, log_dir, model_path = setup_training_args_and_logs(
        game_name="AtariBreakout",
        model_name="atari_breakout_model"
    )
    gym.register_envs(ale_py)
    
    print_training_header("Atari Breakout")
    
    # --- 2. 创建环境 ---
    print("📦 正在创建训练环境...")
    # 训练环境，使用并行化加速
    # make_atari_env 会自动处理大部分预处理工作
    train_env = make_atari_env('ALE/Breakout-v5', n_envs=4, seed=0)
    # VecFrameStack 将连续的4帧图像堆叠起来，让智能体能感知到运动方向
    train_env = VecFrameStack(train_env, n_stack=4)
    print(f"✅ 训练环境创建完成，使用 {4} 个并行环境")
    
    # 单独创建一个用于可视化的环境
    eval_env = make_atari_env('ALE/Breakout-v5', n_envs=1, seed=1)  # 使用不同的seed避免和训练环境完全一样
    eval_env = VecFrameStack(eval_env, n_stack=4)
    # eval_env = gym.make('ALE/Breakout-v5', render_mode='human')
    # eval_env = VecFrameStack(eval_env, n_stack=4)
    print("✅ 可视化环境创建完成")

    # --- 3. 实例化回调函数 ---
    print("🎯 设置可视化回调函数...")
    # 设置每 50,000 步可视化一次（Atari游戏训练时间较长）
    vis_callback = VisualizationCallback(eval_env, eval_freq=50000)
    print("✅ 可视化回调函数设置完成，每50000步展示一次效果")

    # --- 4. 创建或加载 PPO 模型 ---
    if args.resume:
        print("🔄 正在加载现有模型...")
        if not os.path.exists(args.resume):
            print(f"❌ 错误：模型文件不存在: {args.resume}")
            return
        
        model = PPO.load(args.resume, env=train_env, reset_num_timesteps=False)
        print(f"✅ 模型加载成功: {args.resume}")
        print("📊 将继续之前的训练进度")
    else:
        print("🆕 正在创建新的PPO模型...")
        # 使用 CnnPolicy 创建 PPO 模型，注意这里的策略变成了 "CnnPolicy"
        model = PPO(
            "CnnPolicy",
            train_env,
            verbose=1,
            tensorboard_log=log_dir,
            device='auto'
        )
        print("✅ 新PPO模型创建完成")

    print("\n" + "=" * 60)
    print("🏃‍♂️ 开始训练...")
    print("=" * 60)
    
    # 在 learn() 方法中传入 callback
    # Atari 游戏通常需要更长的训练时间，例如几百万步
    model.learn(
        total_timesteps=1_000_000,
        tb_log_name="PPO_with_vis",
        callback=vis_callback
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