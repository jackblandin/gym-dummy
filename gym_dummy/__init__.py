from gym.envs.registration import register

register(
    id='Dummy-v0',
    entry_point='gym_dummy.envs:DummyEnv')
