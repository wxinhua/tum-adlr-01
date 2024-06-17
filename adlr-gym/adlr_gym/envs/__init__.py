from gymnasium.envs.registration import register

register(
    id='MapEnv-v1',
    entry_point='envs.map_env:MapEnv',
    max_episode_steps=200,
)
