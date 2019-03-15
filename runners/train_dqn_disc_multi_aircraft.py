import argparse
import json
import sys
import time

import numpy as np
import tensorflow as tf

import rltools.algos.dqn
import rltools.log
import rltools.util
from rltools.qnet.categorical_qnet import CategoricalQFunction
from madrl_environments.cas.multi_aircraft import *


LARGE_VAL_ARCH = '''[
        {"type": "fc", "n": 256},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 128},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 64},
        {"type": "nonlin", "func": "tanh"}
    ]
    '''

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_iter', type=int, default=4000000)
    parser.add_argument('--target_update_step', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--discount', type=float, default=0.95)
    parser.add_argument('--max_experience_size', type=int, default=20000)
    parser.add_argument('--traj_sim_len', type=int, default=300)
    parser.add_argument('--n_eval_traj', type=int, default=2)
    parser.add_argument('--n_agents', type=int, default=5)

    parser.add_argument('--save_freq', type=int, default=5000)
    parser.add_argument('--log_file_name', type=str, default='default')


    parser.set_defaults(debug=True)

    args = parser.parse_args()

    env = MultiAircraftEnv(n_agents=args.n_agents, continuous_action_space=False)


    q_func = CategoricalQFunction(env.observation_space, env.action_space,
                                      hidden_spec=LARGE_VAL_ARCH,
                                      learning_rate=1e-4,
                                      primary_q_func=None,
                                      dueling=False, 
                                      varscope_name='q_func')

    target_q_func = CategoricalQFunction(env.observation_space, env.action_space,
                                            hidden_spec=LARGE_VAL_ARCH,
                                            learning_rate=1e-4,
                                            primary_q_func=q_func,
                                            dueling=False,
                                            varscope_name='target_q_func')

    def convert_obs(obs):
        return np.array(obs)

    dqn_opt = rltools.algos.dqn.DQN(env=env,
                                q_func=q_func,
                                target_q_func=target_q_func,
                                obsfeat_fn=convert_obs, 
                                target_update_step=args.target_update_step, 
                                batch_size=args.batch_size,
                                discount=args.discount,
                                n_iter=args.n_iter,
                                max_experience_size=args.max_experience_size,
                                traj_sim_len=args.traj_sim_len,
                                n_eval_traj=args.n_eval_traj)

    args_ = {'eps_decay': True,
            'arch': 'LARGE_VAL_ARCH'}
    argstr = json.dumps(args_, separators=(',',':'), indent=2)
    rltools.util.header(argstr)
    if args.log_file_name == 'default':
        log_file_name = 'dqn_train_log_' + time.strftime("%Y%m%d-%H%M%S") + '.h5'
    else:
        log_file_name = args.log_file_name + '.h5'

    log_f = rltools.log.TrainingLog(log_file_name, [('args', argstr)], debug=True)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        dqn_opt.train(sess, log_f, args.save_freq)


if __name__ == '__main__':
    main()
