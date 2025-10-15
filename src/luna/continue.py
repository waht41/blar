# ...（之前的代码，包括定义模型等）
from stable_baselines3 import PPO

# 加载你已经训练到一半的模型
model_path = "./ppo_lunarlander_logs/lunar_lander_model.zip"
model = PPO.load(model_path, env=train_env) # 确保把新的环境对象传进去

print("模型已加载，开始继续训练...")
# reset_num_timesteps=False 确保时间步是连续的，而不是从0开始
model.learn(
    total_timesteps=500_000, # 设置一个更大的总目标
    tb_log_name="PPO_continue_run",
    callback=vis_callback,
    reset_num_timesteps=False # 关键！
)
print("继续训练完成！")

# 保存最终的模型
model.save("./ppo_lunarlander_logs/final_model.zip")