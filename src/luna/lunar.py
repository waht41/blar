from datetime import datetime
import argparse

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import os
import time

from src.callbacks import VisualizationCallback


def main():
    """主函数：训练LunarLander PPO模型"""
    # --- 1. 解析命令行参数 ---
    parser = argparse.ArgumentParser(description='训练LunarLander PPO模型')
    parser.add_argument('--resume', type=str, help='从指定路径恢复训练模型')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 LunarLander PPO 训练程序启动")
    print("=" * 60)
    
    # --- 2. 创建环境和日志目录 ---
    print("📦 正在创建训练环境...")
    # 训练环境，使用并行化加速
    train_env = make_vec_env("LunarLander-v3", n_envs=16)
    print(f"✅ 训练环境创建完成，使用 {16} 个并行环境")
    
    # 单独创建一个用于可视化的环境
    eval_env = gym.make("LunarLander-v3", render_mode="human")
    print("✅ 可视化环境创建完成")
    
    # 设置日志目录
    if args.resume:
        # 如果是从现有模型恢复，使用原模型的日志目录
        model_dir = os.path.dirname(args.resume)
        log_dir = model_dir
        print(f"🔄 从现有模型恢复训练: {args.resume}")
        print(f"📁 使用现有日志目录: {log_dir}")
    else:
        # 创建新的日志目录
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_id = f"{timestamp}_ppo_lunarlander_logs"
        log_dir = f"./logs/{run_id}/"
        os.makedirs(log_dir, exist_ok=True)
        print(f"📁 创建新日志目录: {log_dir}")
    
    model_path = os.path.join(log_dir, "lunar_lander_model.zip")

    # --- 3. 实例化回调函数 ---
    print("🎯 设置可视化回调函数...")
    # 设置每 20,000 步可视化一次
    vis_callback = VisualizationCallback(eval_env, eval_freq=20000)
    print("✅ 可视化回调函数设置完成，每20000步展示一次效果")

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
        model = PPO(
            "MlpPolicy",
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
    model.learn(
        total_timesteps=200_000,
        tb_log_name="PPO_with_vis",
        callback=vis_callback
    )
    
    print("\n" + "=" * 60)
    print("🎉 训练完成！")
    print("=" * 60)

    # --- 5. 保存最终模型 ---
    print("💾 正在保存最终模型...")
    model.save(model_path)
    print(f"✅ 最终模型已保存至: {model_path}")
    print(f"📊 要查看训练日志，请在终端运行: tensorboard --logdir ./logs")
    
    # --- 6. 清理环境 ---
    print("🧹 正在清理环境...")
    train_env.close()
    eval_env.close()
    print("✅ 环境清理完成")
    print("👋 程序结束")


if __name__ == "__main__":
    main()

