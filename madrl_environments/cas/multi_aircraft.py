import copy
import math
from math import pi
import sys

import gym
import numpy as np
import random
from gym import spaces
from gym.utils import colorize, seeding
from six.moves import xrange
import Box2D
from madrl_environments import AbstractMAEnv, Agent

from rltools.util import EzPickle

# Multi-agent settings
MIN_AGENTS = 10
MAX_AGENTS = 60

DEST_THRESHOLD = 100 # in m

# Sensor model parameters
SENSING_RANGE = 1000 # in m
SENSOR_CAPACITY = 4
NMAC_RANGE = 150 # in s

# Agent continuous dynamics properties
ACTION_IND_V = 0
ACTION_IND_TURN = 1
ACTION_DIM = 2
MIN_V = 5 # in m/s
MAX_V = 50 # in m/s
MIN_TURN_RATE = 0 # in rad/s
MAX_TURN_RATE = 10 # in rad/s

# Training settings
DT = 1 # in s
MAX_TIME_STEPS = 2000 # in s
TRAINING_SCENARIOS = ['circle', 'annulus', 'square']

# For training scenario: on circle
MIN_CIRCLE_RADIUS = 3000 # in m 
MAX_CIRCLE_RADIUS = 4000 # in m 

# For training scenario: in annulus
INNER_RADIUS = 2000 # in m 
OUTTER_RADIUS = 4000 # in m 

# For training scenario: in square space
AIRSPACE_WIDTH = 10000 # in m 

# Angle range helper
# wrap an angle in (- pi, pi] 
def norm_angle(angle):
    return (angle + pi) % (2 * pi) - pi

class Aircraft(Agent): 

    def __init__(self, env):
        self.env = env
        self.v = MIN_V
        self.turn_rate = 0
        if self.env.training_mode == 'circle':
            r = self.env.circle_radius
            init_position_angle = random.uniform(- pi, pi)
            self.x = r * np.cos(init_position_angle)
            self.y = r * np.sin(init_position_angle)
            self.dest_x = r * np.cos(init_position_angle + pi)
            self.dest_y = r * np.sin(init_position_angle + pi)

        elif self.env.training_mode == 'annulus':
            init_r = random.uniform(INNER_RADIUS, OUTTER_RADIUS)
            dest_r = random.uniform(INNER_RADIUS, OUTTER_RADIUS)
            init_position_angle = random.uniform(- pi, pi)
            self.x = init_r * np.cos(init_position_angle)
            self.y = init_r * np.sin(init_position_angle)
            self.dest_x = dest_r * np.cos(init_position_angle + pi)
            self.dest_y = dest_r * np.sin(init_position_angle + pi)

        elif self.env.training_mode == 'square':
            self.x = AIRSPACE_WIDTH * random.uniform(0, 1)
            self.y = AIRSPACE_WIDTH * random.uniform(0, 1)
            self.dest_x = AIRSPACE_WIDTH * random.uniform(0, 1)
            self.dest_y = AIRSPACE_WIDTH * random.uniform(0, 1)
            
        self.heading = norm_angle(math.atan2(self.dest_y - self.y, self.dest_x - self.x))
        self.dist_to_dest = np.sqrt((self.dest_y - self.y)**2 + (self.dest_x - self.x)**2)
        self.init_dist_to_dest = self.dist_to_dest
        self.prev_dist_to_dest = self.dist_to_dest

        # init agent obs
        self.obs = [
            self.v / MAX_V, # [0, 1]
            self.turn_rate / MAX_TURN_RATE, # [-1, 1]
            self.dist_to_dest / self.init_dist_to_dest # [0, 1],
            norm_angle(math.atan2(self.dest_y - self.y, self.dest_x - self.x) 
                - self.heading) / pi, # [-1, 1], Angle of destination wrt agent 
        ] + [1] * 4 * self.env.sensor_capacity

        # obs from intruders [dist, angle_wrt_heading, heading_diff, v_int]

        assert len(self.obs) == 4 + 4 * self.env.sensor_capacity

    def apply_action(self, action):
        # Entries of action vector in [-1, 1]
        self.v = (MAX_V - MIN_V) / 2 * (action[ACTION_IND_V] - 1) * MAX_V
        self.turn_rate = MAX_TURN_RATE * action[ACTION_IND_TURN]
        # Update coordinates
        self.x += np.cos(self.heading) * self.v * DT
        self.y += np.sin(self.heading) * self.v * DT
        self.prev_dist_to_dest = self.dist_to_dest
        self.dist_to_dest = np.sqrt((self.dest_y - self.y)**2 + (self.dest_x - self.x)**2)

    def get_observation(self):
        # Update and return self.obs
        if self.env.sensor_mode == 'sector':
            pass # TODO
        elif self.env.sensor_mode == 'closest':

            return self.obs

    def nmac(self):
        intruder_dist_ind = [4 * (i + 1) for i in range(self.env.sensor_capacity)]
        return True if any(np.array(self.obs(intruder_dist_ind)) 
            < NMAC_RANGE / SENSING_RANGE) else False

    def arrival(self):
        return True if self.dist_to_dest < DEST_THRESHOLD else False

    def reward(self):
        reward = 0
        if self.arrival():
            reward += self.env.rew_arrival
        else:
            reward += self.env.rew_closing * (self.prev_dist_to_dest - self.dist_to_dest)

        if self.nmac():
            reward += self.env.rew_nmac

        if np.abs(self.turn_rate) > 0.7 * MAX_TURN_RATE
            reward += self.env.rew_large_turnrate * np.abs(self.turn_rate)

        return reward

    @property
    def observation_space(self):
        return

    @property
    def action_space(self):
        if self.env.continuous_action_space:
            return spaces.Box(low=-1, high=1, shape=(ACTION_DIM,))
        else:
            pass # TODO
    
class MultiAircraftEnv(AbstractMAEnv, EzPickle):

    def __init__(self, 
                 continuous_action_space=True,
                 n_agents=MIN_AGENTS,
                 training_mode='circle', 
                 sensor_mode='closest',
                 sensor_capacity=SENSOR_CAPACITY,
                 max_time_steps=MAX_TIME_STEPS,
                 rew_arrival=15,
                 rew_closing=2.5,
                 rew_nmac=-15,
                 rew_large_turnrate=-0.1):

        self.agents = []
        self.training_mode = training_mode
        self.sensor_mode = sensor_mode
        self.max_time_steps = max_time_steps
        # Reward weights:
        self.rew_arrival = rew_arrival
        self.rew_closing = rew_closing
        self.rew_nmac = rew_nmac
        self.rew_large_turnrate = rew_large_turnrate
        self.setup()

    def setup(self):
        # TODO
        self.reset()

    @property
    def agents(self):
        return self.agents

    def get_param_values(self):
        return self.__dict__

    def reset(self):
        if training_mode == 'circle':
            self.circle_radius = random.choice(range(MIN_CIRCLE_RADIUS, MAX_CIRCLE_RADIUS))
        elif training_mode == 'annulus':

        elif training_mode == 'square':

    def agents_control(self):
        pass
    
    def step(self):
        pass

    def render(self):
        pass


