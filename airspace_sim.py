import sys
sys.path.append('./rllab')
sys.path.append('./rltools')

import argparse

from madrl_environments.cas.multi_aircraft import *

import matplotlib.pyplot as plt

import joblib
import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser()

def main():
	parser.add_argument('--policy', type=str, default='trpo_full_curr_3_passes_ALL_ENV_Tmax_300_PEN_HEAVY_True_rew_nmac_-150_rew_arr_2_run_2/itr_299.pkl')
	args = parser.parse_args()

	tf.reset_default_graph()
	with tf.Session() as sess:
		data = joblib.load('./rllab/data/' + args.policy)
		policy = data['policy']
		env = MultiAircraftEnv(n_agents=15, render_option=True)
		env.reset()
	
		done = False
		while not done:
			actions = []
			for ac in env.aircraft:
				obs = ac.get_observation()
				_, action_info = policy.get_action(obs)
				actions.append(action_info['mean'])
			_, _, done, _ = env.step(np.array(actions))

if __name__ == '__main__':
    main()