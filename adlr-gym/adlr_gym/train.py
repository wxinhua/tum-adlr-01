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
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
import torch.optim as optim
import pygame
import matplotlib.pyplot as plt


device = torch.device("cuda")

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim=5)  
        #self.my_model = MyModel()
        self.my_model = MyModel_test().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.extractor = MyModel_test(self.device)

    def forward(self, observations):
        
        observations = observations.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return self.my_model(observations)
        # observations = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        # features = self.extractor(observations)
        # return features
    


def custom_rmsprop(params, lr=3e-5, **kwargs):
    return optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-5, **kwargs)  



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
        
        fraction = min(1.0, self.num_timesteps / self.max_timesteps)
        
        self.epsilon = self.initial_epsilon + fraction * (self.final_epsilon - self.initial_epsilon)
        return True

    def _on_training_start(self):
        
        self.epsilon = self.initial_epsilon

    def _on_training_end(self):
        
        print(f"Final epsilon: {self.epsilon}")







#env = MapEnv()
env = make_vec_env('MapEnv-v0', n_envs=50)
env = VecNormalize(env, norm_obs=True, norm_reward=True)
env = VecFrameStack(env, n_stack=3)
eval_env = make_vec_env('MapEnv-v0', n_envs=1)
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
eval_env = VecFrameStack(eval_env, n_stack=3)

total_steps = 5000000
change_lr_timestep = 500000
def learning_rate_schedule(progress_remaining):
     
    current_timestep = total_steps * (1 - progress_remaining)
    if current_timestep < change_lr_timestep:
        return 1e-4  # 1e-3
    elif change_lr_timestep <= current_timestep <= int(change_lr_timestep * 1.5):
        return 5e-5
    else:
        return 3e-5

reward_logger = RewardLogger()
epsilon_callback = EpsilonGreedyCallback(max_timesteps=total_steps)

eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/best_model/',
                             log_path='./logs/results/', eval_freq=10000, n_eval_episodes=10,
                             deterministic=True, render=False)



# model = DQN("MlpPolicy",
#     env,learning_rate=learning_rate_schedule,batch_size=256,policy_kwargs={"features_extractor_class": CustomFeatureExtractor},exploration_initial_eps=1.0,
#     exploration_final_eps=0.1,
#     exploration_fraction=0.15,verbose=1, tensorboard_log="./map")


model = PPO("MlpPolicy",
    env,learning_rate=learning_rate_schedule,batch_size=512,policy_kwargs={"features_extractor_class": CustomFeatureExtractor},verbose=1, tensorboard_log="./map",device=device)

# model = DQN.load('logs/best_model/best_model_01.zip')
# model.set_env(env)
# model.policy.features_extractor_class = CustomFeatureExtractor
# model.learning_rate = learning_rate_schedule


model.learn(total_timesteps=total_steps, callback=[eval_callback, reward_logger, epsilon_callback])
#model.save("dqn_model_vec")
model.save("ppo_model_d4")
#env.save("train_vec_normalize_s.pkl")
env.save("train_vec_normalize_d4.pkl")











