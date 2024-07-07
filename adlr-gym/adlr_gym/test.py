import gymnasium as gym
from adlr_gym.envs.map_env import MapEnv  
import matplotlib.pyplot as plt
import pygame
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt



#eval_env = make_vec_env('MapEnv-v0', n_envs=1)
eval_env = MapEnv()

#model = PPO.load('logs/best_model/best_model.zip')
model = PPO.load('ppo_model_v1.zip')

# eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)
# eval_env.training = False
# eval_env.norm_reward = False

clock = pygame.time.Clock()
obs, info = eval_env.reset()
pygame.init()
for _ in range(1000): 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, terminated, truncated, info = eval_env.step(action)

    eval_env.render()
    
    if terminated or truncated:
        obs, info = eval_env.reset()




