import copy
import math
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

# Agent dynamics properties
DT = 1 # in s
MIN_V = 5 # in m/s
MAX_V = 50 # in m/s
MIN_TURN_RATE = 0 # in rad/s
MAX_TURN_RATE = 10 # in rad/s

MAX_TIME_STEPS = 2000
TRAINING_SCENARIOS = ['circle', 'annulus', 'square']

# For training scenario: on circle
MIN_CRICLE_RADIUS = 1000 # in m 
MAX_CIRCLE_RADIUS = 4000 # in m 

# For training scenario: in annulus
INNER_RADIUS = 2000 # in m 
OUTTER_RADIUS = 4000 # in m 

# For training scenario: in square space
AIRSPACE_WIDTH = 10000 # in m 

class Aircraft(Agent):

    def __init__(self, env):
        self.env = env
        self.x = 0
        self.y = 0
        self.heading = 0
        self.v = MIN_V
        self.dest_x = 0
        self.dest_y = 0
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):

        if self.env.training_mode == 'circle':

        elif self.env.training_mode == 'annulus':

        elif self.env.training_mode = 'square':

    def goto_dest(self):
        pass

    def apply_action(self):
        pass

    def get_observation(self):
        
        if self.env.sensor_mode == 'sector':

        elif self.env.sensor_mode == 'closest':


    def nmac(self):
        return

    def reach_dest(self):
        return 

    @property
    def observation_space(self):
        return

    @property
    def action_space(self):
        return
    
class MultiAircraftEnv(AbstractMAEnv, EzPickle):

    def __init__(self, training_mode='circle', 
                 sensor_mode='closest',
                 max_time_steps=MAX_TIME_STEPS):
        self.agents = []
        self.training_mode = training_mode
        self.sensor_mode = sensor_mode
        self.max_time_steps = max_time_steps


    def get_param_values(self):
        return self.__dict__

    def agent_number_control(self):
        pass
    


