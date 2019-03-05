import joblib
import tensorflow as tf
from rllab.sampler.utils import rollout

data_file_path = './rllab/data/experiment_2019_03_05_09_54_11_296673_PST_08c0f/itr_4.pkl'

with tf.Session() as sess:
    data = joblib.load(data_file_path)
    policy = data['policy']
    env = data['env']
    path = rollout(env, policy, max_path_length=1000, 
                    animated=False, speedup=1)


# run_experiment