from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

from madrl_environments.cas.multi_aircraft import *

import matplotlib.pyplot as plt

import joblib
import tensorflow as tf
import numpy as np
# from rllab.misc import tensor_utils
# import time

data_file_path = './rllab/data/test/itr_399.pkl'

XMIN = -1200
XMAX = 1200
YMIN = XMIN
YMAX = XMAX

def vis_slice(env,
        policy,
        resolution=100,
        own_v=45, 
        own_heading=0,
        own_turn_rate=0,
        int_heading=np.deg2rad(270), 
        int_v=15):
    
    intruder = Aircraft(env)

    def get_heat(x, y):
        env.reset()
        env.aircraft[0].x = 0
        env.aircraft[0].y = 0
        env.aircraft[0].dest_x = 2000
        env.aircraft[0].dest_y = 2000
        env.aircraft[0].heading = own_heading
        env.aircraft[0].dist_to_dest = np.sqrt((env.aircraft[0].dest_y)**2 + (env.aircraft[0].dest_x)**2)
        env.aircraft[0].init_dist_to_dest = env.aircraft[0].dist_to_dest
        env.aircraft[0].prev_dist_to_dest = env.aircraft[0].dist_to_dest
        env.aircraft[0].v = own_v
        env.aircraft[0].turn_rate = own_turn_rate
        intruder.x = x
        intruder.y = y
        intruder.heading = int_heading
        env.aircraft.append(intruder)
        obs = env.aircraft[0].get_observation()
        action, action_info = policy.get_action(obs)
        return action, action_info

    acc_map = np.zeros((resolution, resolution))
    turn_rate_map = np.zeros((resolution, resolution))
    x_arr = np.linspace(XMIN, XMAX, resolution)
    y_arr = np.linspace(YMIN, YMAX, resolution)

    # print(get_heat(400, 400))

    for j in range(resolution):
        for i in range(resolution):
            acc_map[j][i] = get_heat(x_arr[i], y_arr[j])[1]['mean'][0]
            turn_rate_map[j][i] = get_heat(x_arr[i], y_arr[j])[1]['mean'][1]
            # print(turn_rate_map[j][i])

    acc_map = np.flipud(acc_map)
    turn_rate_map = np.flipud(turn_rate_map)

    plt.figure()
    plt.imshow(acc_map, cmap="jet", extent=(XMIN, XMAX, YMIN, YMAX))
    plt.colorbar(label='acc')
    plt.figure()
    plt.imshow(turn_rate_map, cmap="jet", extent=(XMIN, XMAX, YMIN, YMAX))
    plt.colorbar(label='turn_rate')
    plt.ion()

def vis(resolution=100,
        own_v=45, 
        own_heading=0,
        own_turn_rate=0,
        int_heading=np.deg2rad(180), 
        int_v=15):
    with tf.Session() as sess:
        data = joblib.load(data_file_path)
        policy = data['policy']
        env = MultiAircraftEnv(n_agents=1)
        vis_slice(env, policy, 
            resolution=resolution,
            own_v=own_v, 
            own_heading=own_heading,
            own_turn_rate=own_turn_rate,
            int_heading=int_heading, 
            int_v=int_v)

        # interact(vis_slice, 
        #     env=fixed(env),
        #     policy=fixed(policy),
        #     resolution=fixed(200),
        #     own_v=(15,45,5),
        #     own_heading=np.deg2rad((-180,180,30)),
        #     own_turn_rate=(0,10,2),
        #     own_dist_to_dest=(0,1,0.2),
        #     own_dist_pos_angle=np.deg2rad((-180,180,30)),
        #     int_heading=np.deg2rad((-180,180,30)), 
        #     int_v=(15,45,5)) # before normalization


