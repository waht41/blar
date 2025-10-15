# CartPole DQN 强化学习项目

## 项目说明
这是一个使用深度Q网络(DQN)算法训练CartPole游戏的强化学习项目。

## 代码结构
重构后的代码包含以下函数：

### 核心函数
- `create_environment(render_mode="human")`: 创建CartPole环境
- `train_model(env, total_timesteps=2000)`: 训练DQN模型
- `load_model(env)`: 加载已保存的模型
- `evaluate_model(model, env, n_episodes=10)`: 评估模型性能
- `demo_model(model, env, episodes=5)`: 可视化演示模型玩游戏
- `main(train_new_model=True)`: 主控制函数

## 使用方法

### 第一次运行（训练新模型）
```python
# 在 main.py 文件末尾修改参数
main(train_new_model=True)
```

### 第二次及以后运行（直接加载模型）
```python
# 在 main.py 文件末尾修改参数
main(train_new_model=False)
```

## 运行模式说明

1. **训练模式** (`train_new_model=True`):
   - 创建新环境
   - 初始化并训练DQN模型
   - 保存模型到 `dqn_cartpole_model.zip`
   - 评估模型性能
   - 演示模型玩游戏

2. **加载模式** (`train_new_model=False`):
   - 创建环境
   - 直接加载已保存的模型
   - 评估模型性能
   - 演示模型玩游戏

## 注意事项
- 如果模型文件不存在，程序会自动切换到训练模式
- 可以通过修改函数参数来自定义训练步数、评估回合数等
- 确保已安装必要的依赖包：`gymnasium`, `stable-baselines3`
