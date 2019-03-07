from madrl_environments.cas.multi_aircraft import *
import numpy as np

n_agents = 2
env = MultiAircraftEnv(n_agents=n_agents, render_option=True, constant_n_agents=True)
env.reset()

for _ in range(MAX_TIME_STEPS):
    # a = np.array([env.agents[0].action_space.sample() for _ in range(n_agents)])
    a = np.array([0,0] * n_agents)
    o, r, done, _ = env.step(a)
    # print("\nStep:", i)
    # print("Obs:", o)
    # print("Rewards:", r)
    #print "Term:", done
    if done:
        break
