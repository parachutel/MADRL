import sys
sys.path.append('../../../rllab')
sys.path.append('../../../rltools')
sys.path.append('../../../madrl_environments')

from madrl_environments.cas.multi_aircraft import *

import argparse
import matplotlib.pyplot as plt

import joblib
import tensorflow as tf
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_eval', type=int, default=1)
    parser.add_argument('--policy_file', type=str, default='')
    parser.add_argument('--n_agents', type=int, default=10)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--eval_time_steps', type=int, default=500)
    args = parser.parse_args()

    print('Evaluating policy: {}'.format(args.policy_file))
    print('Number of agents = {}'.format(args.n_agents))

    policy_file = '../../../rllab/data/' + args.policy_file
    tf.reset_default_graph()
    with tf.Session() as sess:
        if args.policy_file != 'none':
            policy_file_data = joblib.load(policy_file)
            policy = policy_file_data['policy']

        env = MultiAircraftEnv(n_agents=args.n_agents, 
                                render_option=args.render,
                                constant_n_agents=True)
        nmac_per_time_step_per_ac = []
        for i_eval in range(args.n_eval):
            env.reset()
            t = 0
            nmac_count = 0
            while t < args.eval_time_steps:
                actions = []
                for ac in env.aircraft:
                    if ac.nmac():
                        nmac_count += 1
                    obs = ac.get_observation()
                    if args.policy_file != 'none':
                        _, action_info = policy.get_action(obs)
                        actions.append(action_info['mean'])
                    else:
                        actions.append([0] * ACTION_DIM)
                
                env.step(np.array(actions))
                env.n_agents_control()
                t += 1
    
            nmac_per_time_step_per_ac.append(nmac_count / 2 / args.eval_time_steps / args.n_agents)
            print('Eval #{}, NMAC per time step per agent = {}'.format(i_eval+1, nmac_per_time_step_per_ac[-1]))

        print('Evals end, NMAC per time step per agent = {}'.format(np.mean(nmac_per_time_step_per_ac)))

if __name__ == '__main__':
    main()