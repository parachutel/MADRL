import joblib
import rllab
import tensorflow as tf
import json

data_file_path = './rllab/data/experiment_2019_03_01_17_04_37_793639_PST_d1c5e/itr_4.pkl'

with tf.Session() as sess:
	data = joblib.load(data_file_path)
	print(data['policy'])


# run_experiment