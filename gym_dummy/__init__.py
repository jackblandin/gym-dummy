from gym.envs.registration import register

register(
    id='GreaterThanZero-v0',
    entry_point='gym_dummy.envs.fobs:GreaterThanZeroEnv')

register(
    id='NotXOR-v0',
    entry_point='gym_dummy.envs.fobs:NotXOREnv')

register(
    id='TwoInARow-v0',
    entry_point='gym_dummy.envs.pobs:TwoInARowEnv')
