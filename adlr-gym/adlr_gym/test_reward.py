import gymnasium as gym
import numpy as np
import pygame
from adlr_gym.envs.map_env import MapEnv # 请将my_env_module替换为你的环境模块的实际名称

# 定义动作
ACTIONS = {
    pygame.K_UP: 0,
    pygame.K_DOWN: 1,
    pygame.K_LEFT: 2,
    pygame.K_RIGHT: 3,
    pygame.K_SPACE: 4  # Idle action
}

# 初始化环境
env = MapEnv()
obs, info = env.reset()

# 设置Pygame窗口
pygame.init()
window_size = 1000
window = pygame.display.set_mode((window_size, window_size))
pygame.display.set_caption("RL Agent Control")
clock = pygame.time.Clock()
fps = 10

# 主循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    action = None
    for key in ACTIONS:
        if keys[key]:
            action = ACTIONS[key]
            break

    if action is not None:
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}")

        if terminated or truncated:
            print("Episode finished!")
            obs, info = env.reset()

    # 渲染环境
    env.render()
    clock.tick(fps)

pygame.quit()
