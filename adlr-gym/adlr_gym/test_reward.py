import gymnasium as gym
import numpy as np
import pygame
from adlr_gym.envs.map_env import MapEnv  # 请将my_env_module替换为你的环境模块的实际名称

# 定义动作
ACTIONS = {
    pygame.K_UP: 0,
    pygame.K_DOWN: 1,
    pygame.K_LEFT: 2,
    pygame.K_RIGHT: 3,
    pygame.K_SPACE: 4  # Idle action
}


env = MapEnv()
obs, info = env.reset()


pygame.init()
window_size = 1000
window = pygame.display.set_mode((window_size, window_size))
pygame.display.set_caption("RL Agent Control")
clock = pygame.time.Clock()
fps = 5


running = True
while running:
    action = None
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in ACTIONS:
                action = ACTIONS[event.key]
    
    if action is not None:
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}")

        if terminated or truncated:
            print("Episode finished!")
            obs, info = env.reset()

    env.render()
    clock.tick(fps)

pygame.quit()