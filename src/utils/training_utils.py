"""
训练工具模块
包含解析命令行参数和生成日志路径的通用函数
"""
import argparse
import os
from datetime import datetime
from typing import Tuple, Optional


def setup_training_args_and_logs(
    game_name: str,
    model_name: str = "model",
    additional_args: Optional[list] = None
) -> Tuple[argparse.Namespace, str, str]:
    """
    设置训练参数和日志路径的通用函数
    
    Args:
        game_name: 游戏名称，用于生成日志目录名
        model_name: 模型文件名（不含扩展名）
        additional_args: 额外的命令行参数列表，格式为 [(name, kwargs), ...]
    
    Returns:
        Tuple[args, log_dir, model_path]: 
        - args: 解析后的命令行参数
        - log_dir: 日志目录路径
        - model_path: 模型文件完整路径
    """
    # 创建参数解析器
    parser = argparse.ArgumentParser(description=f'训练{game_name} PPO模型')
    parser.add_argument('--resume', type=str, help='从指定路径恢复训练模型')
    
    # 添加额外参数
    if additional_args:
        for arg_name, arg_kwargs in additional_args:
            parser.add_argument(arg_name, **arg_kwargs)
    
    args = parser.parse_args()
    
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
        run_id = f"{timestamp}_ppo_{game_name.lower()}_logs"
        log_dir = f"./logs/{run_id}/"
        os.makedirs(log_dir, exist_ok=True)
        print(f"📁 创建新日志目录: {log_dir}")
    
    # 生成模型文件路径
    model_path = os.path.join(log_dir, f"{model_name}.zip")
    
    return args, log_dir, model_path


def print_training_header(game_name: str):
    """
    打印训练程序启动头部信息
    
    Args:
        game_name: 游戏名称
    """
    print("=" * 60)
    print(f"🚀 {game_name} PPO 训练程序启动")
    print("=" * 60)


def print_training_footer(log_dir: str):
    """
    打印训练完成后的信息
    
    Args:
        log_dir: 日志目录路径
    """
    print("\n" + "=" * 60)
    print("🎉 训练完成！")
    print("=" * 60)
    print(f"📊 要查看训练日志，请在终端运行: tensorboard --logdir ./logs")
