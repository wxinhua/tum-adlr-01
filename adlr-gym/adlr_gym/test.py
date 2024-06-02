import gymnasium as gym
from adlr_gym.envs.map_env import MapEnv  
import matplotlib.pyplot as plt
import pygame
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
""" env = MapEnv()
observation = env.reset()
plt.figure()
#render = env.render()
#plt.figure()
plt.imshow(observation)
plt.show() """

""" env = MapEnv()
observation = env.reset()
for _ in range(50):  
    action = env.action_space.sample()  
    observation, reward, terminated, truncated, info = env.step(action)  
    env.render()  # 使用 Pygame 渲染环境状态
    #print(f"Current Position: {env.current_position}, Action Taken: {action}, Next Position: {env._move_robot(action)}")

    if terminated:
        break  # 如果达到终点或任何终止条件，退出循环

    for event in pygame.event.get():  
        if event.type == pygame.QUIT:  # 
            done = True

env.close()  # 关闭环境和 Pygame 窗口 """

#env = gym.make('MapEnv-v0')
#env = make_vec_env(lambda: env, n_envs=1)
#env = DummyVecEnv([lambda: MapEnv()])
#env = VecTransposeImage(env)
env = MapEnv()
# 加载模型
model = DQN.load("dqn_model_v1")

# 评估模型
""" mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"平均奖励: {mean_reward}, 奖励标准差: {std_reward}")
 """
# 在环境中展示模型
obs = env.reset()
pygame.init()
for _ in range(100): 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    action = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, info = env.step(action)
    env.render()
    #env.envs[0].render()
    if terminated:
        obs = env.reset()
