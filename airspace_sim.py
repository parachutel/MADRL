import sys
sys.path.append('./rllab')
sys.path.append('./rltools')

from madrl_environments.cas.multi_aircraft import *

import matplotlib.pyplot as plt

import joblib
import tensorflow as tf
import numpy as np

# data_file_path = './rllab/data/test/itr_399_Mar5_work_for_circle.pkl' # 399 Mar 5 works for circle
data_file_path = './rllab/data/test/itr_99.pkl'

tf.reset_default_graph()
with tf.Session() as sess:
	data = joblib.load(data_file_path)
	policy = data['policy']
	env = MultiAircraftEnv(n_agents=10, render_option=True)
	env.reset()

	done = False
	while not done:
		actions = []
		for ac in env.aircraft:
			obs = ac.get_observation()
			_, action_info = policy.get_action(obs)
			actions.append(action_info['mean'])
		_, _, done, _ = env.step(np.array(actions))