import gymnasium as gym
from adlr_gym.envs.map_env import MapEnv  
import matplotlib.pyplot as plt
import pygame
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack

eval_env = make_vec_env('MapEnv-v0', n_envs=1)
eval_env = VecNormalize.load("train_vec_normalize_d4.pkl", eval_env)
# eval_env = VecNormalize.load("train_vec_normalize_s.pkl", eval_env)
eval_env = VecFrameStack(eval_env, n_stack=3)
model = PPO.load('ppo_model_d4.zip')
# model = PPO.load('ppo_model_s.zip')
# 
clock = pygame.time.Clock()
obs = eval_env.reset()
pygame.init()

# Initialize counters
success_count = 0
total_count = 0

# Set up Pygame display
screen = pygame.display.set_mode((1000, 1000))
pygame.display.set_caption("RL Agent Performance")
font = pygame.font.SysFont(None, 36)

# Initial text render
text = font.render(f'Success: {success_count} / Total: {total_count}', True, (255, 255, 255))
text_rect = text.get_rect(topleft=(10, 10))
screen.blit(text, text_rect)
pygame.display.update(text_rect)

for _ in range(1000): 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = eval_env.step(action)


    eval_env.envs[0].render()
    
    # Update counts
    if done or info[0].get('TimeLimit.truncated', False):
        total_count += 1
        if rewards > 0:  # Adjust the condition based on what constitutes a success in your env
            success_count += 1
        obs = eval_env.reset()
    

    screen.fill((0, 0, 0))  
    new_text = font.render(f'Success: {success_count} / Total: {total_count}', True, (255, 255, 255))
    text_rect = new_text.get_rect(topleft=(10, 10))
    screen.blit(new_text, text_rect)
    pygame.display.update(text_rect)  
    if total_count > 0 :
        print(f"Success rate: {success_count / total_count}")
    


pygame.quit()
