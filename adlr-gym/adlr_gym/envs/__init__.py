from adlr_gym.envs.map_env import MapEnv
from gymnasium.envs.registration import register

register(
    id='MapEnv-v1',
    entry_point='envs.map_env:MapEnv',
    max_episode_steps=200,
)
