import ale_py # ale_py 需要被 import 一次，让 gym 能够发现 Atari 环境
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
import os
import yaml
from datetime import datetime
from typing import Callable  # 导入 Callable 用于定义学习率调度
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
    WarpFrame,
)

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

def make_single_atari_env(env_id: str, seed: int, noop_max: int = 30, skip: int = 4, frame_size: int = 84, **kwargs) -> gym.Env:
    """
    为 Atari 创建一个经过标准预处理封装的单个环境。
    这个版本包含了 Monitor wrapper 用于日志记录。
    :param env_id: 环境ID
    :param seed: 随机种子
    :param noop_max: NoopResetEnv的最大随机操作数
    :param skip: MaxAndSkipEnv的跳帧数
    :param frame_size: WarpFrame的帧大小
    :param kwargs: 其他环境参数
    """
    env = gym.make(env_id, **kwargs)
    env.action_space.seed(seed)

    # 关键：在应用其他 Wrapper 之前或之后（通常是较早）添加 Monitor
    # Monitor 需要在 EpisodicLifeEnv 之前，以正确记录每个"生命"的信息
    env = Monitor(env)

    env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=skip)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env, width=frame_size, height=frame_size)
    env = ClipRewardEnv(env)

    env.reset(seed=seed)
    return env

def load_config(config_path: str = "src/atari/config.yaml") -> dict:
    """
    加载YAML配置文件
    :param config_path: 配置文件路径
    :return: 配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        print(f"✅ 配置文件加载成功: {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ 错误：配置文件不存在: {config_path}")
        raise
    except yaml.YAMLError as e:
        print(f"❌ 错误：YAML文件格式错误: {e}")
        raise

def log_config_to_file(config: dict, log_dir: str):
    """
    将配置信息写入日志文件
    :param config: 配置字典
    :param log_dir: 日志目录
    """
    config_log_path = os.path.join(log_dir, "config.yaml")
    try:
        with open(config_log_path, 'w', encoding='utf-8') as file:
            yaml.dump(config, file, default_flow_style=False, allow_unicode=True, indent=2)
        print(f"✅ 配置信息已保存到日志: {config_log_path}")
    except Exception as e:
        print(f"⚠️ 警告：无法保存配置到日志文件: {e}")

def main():
    """主函数：训练Atari Breakout PPO模型"""
    # --- 0. 加载配置文件 ---
    config = load_config()
    
    # --- 1. 解析命令行参数和设置日志目录 ---
    args, log_dir, model_path = setup_training_args_and_logs(
        game_name=config['environment']['game_name'],
        model_name="atari_breakout_model"
    )
    
    # 创建基于tb_log_name的模型保存目录
    model_save_config = config.get('model_save', {})

    # 在log_dir下创建tb_log_name子目录
    model_save_dir = os.path.join(log_dir)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 生成模型文件名
    model_prefix = model_save_config.get('model_prefix', 'atari_breakout_model')
    if model_save_config.get('add_timestamp', True):
        timestamp_format = model_save_config.get('timestamp_format', '%Y%m%d_%H%M%S')
        timestamp = datetime.now().strftime(timestamp_format)
        model_filename = f"{model_prefix}_{timestamp}.zip"
    else:
        model_filename = f"{model_prefix}.zip"
    
    model_path = os.path.join(model_save_dir, model_filename)
    
    print(f"📁 模型将保存到: {model_path}")
    
    # 将配置信息写入日志
    if config.get('logging', {}).get('log_config', True):
        log_config_to_file(config, log_dir)

    # gym.register_envs(ale_py) # 这行是不需要的，只要 ale_py 被导入，gym 就会自动注册环境

    print_training_header("Atari Breakout")

    # --- 2. 创建环境 ---
    print("📦 正在创建训练环境...")
    # 使用 make_atari_env 会自动应用一系列关键的 Wrapper，例如帧跳过(Frame Skipping)
    env_config = config['environment']
    wrapper_config = config['wrappers']
    
    n_envs = env_config['n_envs']
    env_id = env_config['env_id']
    eval_n_envs = env_config['eval_n_envs']
    seed = env_config['seed']
    
    env_fns = [lambda i=i: make_single_atari_env(env_id, seed=i, 
                                               noop_max=wrapper_config['noop_max'],
                                               skip=wrapper_config['skip'],
                                               frame_size=wrapper_config['frame_size']) 
              for i in range(n_envs)]
    train_env = SubprocVecEnv(env_fns)
    # VecFrameStack 将连续的4帧图像堆叠起来，让智能体能感知到运动方向
    train_env = VecFrameStack(train_env, n_stack=4)
    print(f"✅ 训练环境创建完成，使用 {n_envs} 个并行环境")

    eval_env = make_atari_env(env_id, n_envs=eval_n_envs, seed=seed)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    print("✅ 可视化环境创建完成")

    # --- 3. 实例化回调函数 ---
    print("🎯 设置回调函数...")
    callback_config = config['callbacks']
    training_config = config['training']
    
    callbacks_list = []
    
    # 可视化回调
    if callback_config['visualization']['enabled']:
        vis_callback = VisualizationCallback(
            eval_env, 
            eval_freq=callback_config['visualization']['eval_freq']
        )
        callbacks_list.append(vis_callback)
        print("✅ 可视化回调已启用")
    
    # 性能监控回调
    if callback_config['performance']['enabled']:
        performance_callback = PerformanceCallbackWithTqdm(
            verbose=callback_config['performance']['verbose']
        )
        callbacks_list.append(performance_callback)
        print("✅ 性能监控回调已启用")
    
    # 组合多个回调函数
    if callbacks_list:
        callbacks = CallbackList(callbacks_list)
        print(f"✅ 回调函数设置完成，共启用 {len(callbacks_list)} 个回调")
    else:
        callbacks = None
        print("⚠️ 警告：未启用任何回调函数")

    # --- 4. 定义超参数并创建或加载 PPO 模型 ---

    # 从配置文件读取PPO超参数
    ppo_config = config['ppo']
    model_config = config['model']
    
    learning_rate = ppo_config['learning_rate']
    ppo_params = {
        'n_steps': ppo_config['n_steps'],  # 增加每次更新收集的样本数，以获得更稳定的梯度估计
        'batch_size': ppo_config['batch_size'],  # 增加 mini-batch 的大小
        'n_epochs': ppo_config['n_epochs'],  # 减少 epoch 数量，防止在当前数据上过拟合
        'gamma': ppo_config['gamma'],  # 折扣因子
        'gae_lambda': ppo_config['gae_lambda'],  # GAE-Lambda 参数
        'clip_range': ppo_config['clip_range'],  # 减小 clip_range，限制策略更新幅度，提升稳定性
        'ent_coef': ppo_config['ent_coef'],  # 熵系数，鼓励探索
        'vf_coef': ppo_config['vf_coef'],  # 价值函数系数
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
            model_config['policy'],
            train_env,
            verbose=training_config['verbose'],
            tensorboard_log=log_dir if model_config['tensorboard_log'] else None,
            device=model_config['device'],
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

    # 从配置文件读取总训练步数
    TOTAL_TIMESTEPS = training_config['total_timesteps']
    logging_config = config['logging']

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name=logging_config['tb_log_name'],
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