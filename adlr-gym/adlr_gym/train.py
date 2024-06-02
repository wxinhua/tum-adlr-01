import gymnasium as gym
from adlr_gym.envs.map_env import MapEnv
from network import MyModel  
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import torch.optim as optim

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim=5)  
        self.my_model = MyModel()

    def forward(self, observations):
        return self.my_model(observations)
    


def custom_rmsprop(params, lr=3e-5, **kwargs):
    return optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-5, **kwargs)  



# 训练总步数
total_steps = 20000



env = gym.make('MapEnv-v0', render_mode="human")
env = make_vec_env('MapEnv-v0', n_envs=1)


model = DQN(
    "MlpPolicy",
    env,
    learning_rate=3e-5,
    batch_size=32,
    buffer_size=4000,
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
)
model.learn(total_timesteps=total_steps)






mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
print(f"mean reward: {mean_reward}, std reward: {std_reward}")

model.save("dqn_model_v1")
















