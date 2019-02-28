from runners import RunnerParser
from runners.curriculum import Curriculum

from madrl_environments.cas.multi_aircraft import MultiAircraftEnv
from madrl_environments import StandardizedEnv, ObservationBuffer

# yapf: disable
ENV_OPTIONS = [
    ('n_agents', int, 5, ''),
    ('position_noise', float, 1e-3, ''),
    ('angle_noise', float, 1e-3, ''),
    ('speed_noise', float, 1e-3, ''),
    ('reward_mech', str, 'local', ''),
    ('rew_arrival', float, 1.0, ''),
    ('rew_closing', float, -100.0, ''),
    ('rew_nmac', float, -100.0, ''),
    ('rew_large_turnrate', float, -0.1, ''),
    ('buffer_size', int, 1, ''),
    ('one_hot', int, 0, ''),
    ('curriculum', str, None, ''),
]
# yapf: enable


def main(parser):
    mode = parser._mode
    args = parser.args
    env_config = dict(n_agents=args.n_agents, 
                      speed_noise=args.speed_noise, 
                      position_noise=args.position_noise, 
                      angle_noise=args.angle_noise, 
                      reward_mech=args.reward_mech, 
                      rew_arrival=args.rew_arrival, 
                      rew_closing=args.rew_closing, 
                      rew_nmac=args.rew_nmac, 
                      rew_large_turnrate=args.rew_large_turnrate, 
                      one_hot=bool(args.one_hot))

    env = MultiAircraftEnv(**env_config)
    if args.buffer_size > 1:
        env = ObservationBuffer(env, args.buffer_size)

    if mode == 'rllab':
        from runners.rurllab import RLLabRunner
        run = RLLabRunner(env, args)
    elif mode == 'rltools':
        from runners.rurltools import RLToolsRunner
        run = RLToolsRunner(env, args)
    else:
        raise NotImplementedError()

    if args.curriculum:
        curr = Curriculum(args.curriculum)
        run(curr)
    else:
        run()


if __name__ == '__main__':
    main(RunnerParser(ENV_OPTIONS))