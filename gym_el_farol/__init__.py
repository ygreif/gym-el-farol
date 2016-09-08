from gym.envs.registration import register

register(
    id='ElFarolEnv-v0',
    entry_point='gym.envs.multi_agent:ElFarolEnv',
    timestep_limit=200,
    local_only=True
)
