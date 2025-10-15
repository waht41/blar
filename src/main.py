import time
import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# 模型文件路径
MODEL_PATH = "dqn_cartpole_model.zip"

def create_environment(render_mode="human"):
    """
    创建 CartPole 环境
    
    Args:
        render_mode (str): 渲染模式，默认为 "human" 显示可视化窗口
    
    Returns:
        gym.Env: CartPole 环境实例
    """
    print("--> 正在创建 CartPole 环境...")
    env = gym.make("CartPole-v1", render_mode=render_mode)
    return env

def train_model(env, total_timesteps=2000):
    """
    训练 DQN 模型
    
    Args:
        env: 训练环境
        total_timesteps (int): 训练步数
    
    Returns:
        DQN: 训练好的模型
    """
    print("--> 正在初始化 DQN 模型...")
    model = DQN("MlpPolicy", env, verbose=1)
    
    print("--> 开始训练模型...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    print(f"--> 训练完成，模型已保存至 {MODEL_PATH}")
    model.save(MODEL_PATH)
    
    return model

def load_model(env):
    """
    加载已保存的模型
    
    Args:
        env: 环境实例
    
    Returns:
        DQN: 加载的模型
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件 {MODEL_PATH} 不存在，请先训练模型")
    
    print(f"--> 正在加载模型 {MODEL_PATH}...")
    model = DQN.load(MODEL_PATH, env=env)
    return model

def evaluate_model(model, env, n_episodes=10):
    """
    评估模型性能
    
    Args:
        model: 要评估的模型
        env: 评估环境
        n_episodes (int): 评估回合数
    
    Returns:
        tuple: (平均奖励, 标准差)
    """
    print("--> 正在评估模型性能...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_episodes)
    print(f"--> 评估结果: 平均奖励 = {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward

def demo_model(model, env, episodes=5):
    """
    可视化演示模型玩游戏
    
    Args:
        model: 要演示的模型
        env: 演示环境
        episodes (int): 演示回合数
    """
    print(f"--> 开始可视化AI玩游戏的过程 (共 {episodes} 局)...")
    
    for ep in range(episodes):
        # 重置环境，开始新的一局
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # 使用模型来预测下一步的最佳动作
            action, _states = model.predict(obs, deterministic=True)
            
            # 在环境中执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 累加奖励
            episode_reward += reward
            
            # 检查游戏是否结束
            if terminated or truncated:
                done = True
        
        print(f"  第 {ep + 1} 局结束, 本局得分: {episode_reward}")
        time.sleep(1)  # 暂停一下，方便观察

def main(train_new_model=True):
    """
    主函数
    
    Args:
        train_new_model (bool): 是否训练新模型，False 则直接加载已有模型
    """
    # 创建环境
    env = create_environment()
    
    try:
        if train_new_model or not os.path.exists(MODEL_PATH):
            # 训练新模型
            model = train_model(env)
        else:
            # 加载已有模型
            model = load_model(env)
        
        # 评估模型
        evaluate_model(model, env)
        
        # 演示模型
        demo_model(model, env)
        
    finally:
        # 关闭环境
        env.close()
        print("--> 演示结束。")

if __name__ == "__main__":
    # 可以通过修改这个参数来控制是否训练新模型
    # True: 训练新模型 (第一次运行或想重新训练)
    # False: 直接加载已有模型 (第二次及以后运行)
    main(train_new_model=False)