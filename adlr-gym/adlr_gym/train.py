import gymnasium as gym
from adlr_gym.envs.map_env import MapEnv
from network import MyModel  
import torch
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env


import torch.optim as optim
import pygame
import matplotlib.pyplot as plt

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim=5)  
        self.my_model = MyModel()

    def forward(self, observations):
        return self.my_model(observations)
    


def custom_rmsprop(params, lr=3e-5, **kwargs):
    return optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-5, **kwargs)  

from stable_baselines3.common.callbacks import BaseCallback

class RewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLogger, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        if done := self.locals.get('dones'):
            self.rewards.append(self.locals.get('rewards')[0])
        return True





total_steps = 200000


env = MapEnv()


reward_logger = RewardLogger()

# 设置评估回调以保存最优模型
eval_callback = EvalCallback(env, best_model_save_path='./logs/best_model/',
                             log_path='./logs/results/', eval_freq=10000, n_eval_episodes=10,
                             deterministic=True, render=False)

""" model = DQN(
    "MlpPolicy",
    env,
    learning_rate=3e-5,
    batch_size=32,
    buffer_size=5000,
    learning_starts=1000,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    optimize_memory_usage=False,
    policy_kwargs={"features_extractor_class": CustomFeatureExtractor},
    exploration_initial_eps=1.0,
    exploration_final_eps=0.1,
    exploration_fraction=0.1,
    target_update_interval=10000,
    verbose=1
) """
model = PPO("MlpPolicy",
    env,batch_size=256,policy_kwargs={"features_extractor_class": CustomFeatureExtractor},verbose=1, tensorboard_log="./map")
model.learn(total_timesteps=total_steps, callback=[eval_callback, reward_logger])

# 绘制回报曲线
plt.plot(reward_logger.rewards)
plt.xlabel('Steps')
plt.ylabel('Rewards')
plt.title('Training Rewards')
plt.show()



# rewards = reward_callback.rewards


# rewards = [reward for sublist in rewards for reward in sublist]


# plt.plot(rewards)
# plt.xlabel('Steps')
# plt.ylabel('Reward')
# plt.title('Reward Over Time')
# plt.show()





# """ mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
# print(f"mean reward: {mean_reward}, std reward: {std_reward}") """

# model.save("dqn_model_v1")
# del model # remove to demonstrate saving and loading
# model = DQN.load('dqn_model_v1.zip')

# obs, info = env.reset()
# pygame.init()
# for _ in range(1000): 
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             exit()
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, terminated, truncated, info = env.step(action)
#     env.render()
    
#     if terminated or truncated:
#         obs, info = env.reset()













