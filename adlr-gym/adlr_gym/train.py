import gymnasium as gym
from adlr_gym.envs.map_env import MapEnv
from network import MyModel  
from network_test import MyModel_test
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
        #self.my_model = MyModel()
        self.my_model = MyModel_test()

    def forward(self, observations):
        return self.my_model(observations)
    


def custom_rmsprop(params, lr=3e-5, **kwargs):
    return optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-5, **kwargs)  

from stable_baselines3.common.callbacks import BaseCallback

class RewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLogger, self).__init__(verbose)
        self.rewards = []
        self.total_steps = 0

    def _on_step(self) -> bool:
        # Increment the total steps
        self.total_steps += 1
        # Append reward and current step
        self.rewards.append((self.total_steps, self.locals['rewards'][0]))
        return True



class EpsilonGreedyCallback(BaseCallback):
    def __init__(self, max_timesteps, initial_epsilon=1.0, final_epsilon=0.1, verbose=0):
        super(EpsilonGreedyCallback, self).__init__(verbose)
        self.max_timesteps = max_timesteps
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon = initial_epsilon

    def _on_step(self) -> bool:
        # 计算当前训练步数占总步数的比例
        fraction = min(1.0, self.num_timesteps / self.max_timesteps)
        # 线性减少 epsilon
        self.epsilon = self.initial_epsilon + fraction * (self.final_epsilon - self.initial_epsilon)
        return True

    def _on_training_start(self):
        # 在训练开始时重置 epsilon
        self.epsilon = self.initial_epsilon

    def _on_training_end(self):
        # 在训练结束时打印最终的 epsilon
        print(f"Final epsilon: {self.epsilon}")


total_steps = 400000


env = MapEnv()


reward_logger = RewardLogger()
epsilon_callback = EpsilonGreedyCallback(max_timesteps=total_steps)
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
# model = DQN("MlpPolicy",
#     env,batch_size=32,policy_kwargs={"features_extractor_class": CustomFeatureExtractor},verbose=1, tensorboard_log="./map")

model = PPO("MlpPolicy",
    env,batch_size=256,learning_rate=3e-5, policy_kwargs={"features_extractor_class": CustomFeatureExtractor},verbose=1, tensorboard_log="./map")
#model.learn(total_timesteps=total_steps, callback=[eval_callback, reward_logger, epsilon_callback])
model.learn(total_timesteps=total_steps, callback=[eval_callback, epsilon_callback])
#model.save("dqn_model_v1")
model.save("ppo_model_v1")
# 绘制回报曲线
# steps, rewards = zip(*reward_logger.rewards)
# plt.plot(steps, rewards)
# plt.xlabel('Steps')
# plt.ylabel('Rewards')
# plt.title('Training Rewards')
# plt.show()

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")

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













