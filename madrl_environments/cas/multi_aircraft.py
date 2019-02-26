import copy
import math
import sys

import gym
import numpy as np
from gym import spaces
from gym.utils import colorize, seeding
from six.moves import xrange
import Box2D
from Box2D.b2 import (circleShape, contactListener, edgeShape, fixtureDef, 
                      polygonShape, revoluteJointDef)
from madrl_environments import AbstractMAEnv, Agent

from rltools.util import EzPickle

# Multi-agent settings
MIN_AGENTS = 10
MAX_AGENTS = 60

# Agent dynamics properties
MIN_V = 5
MAX_V = 50
MIN_TURN_RATE = 0
MAX_TURN_RATE = 10

# For training scenario: on circle
MIN_CRICLE_RADIUS = 1000
MAX_CIRCLE_RADIUS = 4000

# For training scenario: in annulus
INNER_RADIUS = 2000
OUTTER_RADIUS = 4000

# For training scenario: in square space
AIRSPACE_WIDTH = 10000
