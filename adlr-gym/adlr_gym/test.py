import gymnasium as gym
from adlr_gym.envs.map_env import MapEnv  
import matplotlib.pyplot as plt
import pygame
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack

import matplotlib.pyplot as plt



eval_env = make_vec_env('MapEnv-v0', n_envs=1)
#eval_env = MapEnv()
eval_env = VecNormalize.load("train_vec_normalize_d3.pkl", eval_env)
eval_env = VecFrameStack(eval_env, n_stack=3)
#model = PPO.load('logs/best_model/best_model.zip')
model = PPO.load('ppo_model_d3.zip')

# eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)
# eval_env.training = False
# eval_env.norm_reward = False

clock = pygame.time.Clock()
obs = eval_env.reset()
pygame.init()
for _ in range(1000): 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    action, _states = model.predict(obs, deterministic=True)
    #obs, rewards, terminated, truncated, info = eval_env.step(action)
    obs, rewards, done, info = eval_env.step(action)
    eval_env.envs[0].render()
    
    #if terminated or truncated:
    if done:
        obs = eval_env.reset()




