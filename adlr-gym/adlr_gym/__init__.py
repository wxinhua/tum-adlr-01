from gymnasium.envs.registration import register

register(
    id='MapEnv-v0',
    entry_point='adlr_gym.envs.map_env:MapEnv',
)

