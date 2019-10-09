from __future__ import division

from pyorca import *
import numpy as np
from numpy import array, rint, linspace, pi, cos, sin
import random
import math

from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib import rcParams, rc
rcParams["figure.figsize"] = [15, 8]
rc('text', usetex=True)
rc("font", family="Times New Roman")
rc("font", size=16)

ANIM = False
LIMIT = True

TAKEOFF_RATE = 60
# 5, 10, 20, 40, 50, 60
n_frames = 5000



SENSING_RANGE = 1000
AIRSPACE_WIDTH = 10000
NMAC_DIST = 150

N_AGENTS = 30 # initial
DEST_THRESHOLD = 100
# RADIUS = NMAC_DIST / 2
RADIUS = 150
MAX_SPEED = 50
if LIMIT:
    MAX_ACC = 2 # 0.2g, from Airbus Vahanna project
    MAX_TURN = np.deg2rad(10)
else:
    MAX_ACC = 10 # 0.2g, from Airbus Vahanna project
    MAX_TURN = np.deg2rad(20)
DT = 1
TAU = DT



def norm_angle(angle):
    return (angle + pi) % (2 * pi) - pi # wrap an angle in (- pi, pi] 

def rand_init_agent():
    x = AIRSPACE_WIDTH * random.uniform(0, 1)
    y = AIRSPACE_WIDTH * random.uniform(0, 1)
    pos = (x, y)
    dest_x = AIRSPACE_WIDTH * random.uniform(0, 1)
    dest_y = AIRSPACE_WIDTH * random.uniform(0, 1)
    dest = (dest_x, dest_y)
    dest_heading = norm_angle(math.atan2(dest_y - y, dest_x - x))
    vel = [MAX_SPEED * np.cos(dest_heading), MAX_SPEED * np.sin(dest_heading)]
    pref_vel = vel
    agent = Agent(pos, dest, vel, RADIUS, MAX_SPEED, pref_vel)
    return agent

def arrival(agent):
    d = agent.position - agent.destination
    dist_to_dest = np.sqrt(np.dot(d, d))
    return True if dist_to_dest < DEST_THRESHOLD else False

class Sim_Stats(object):
    def __init__(self):
        self.total_NMACs = 0
        self.total_flight_hours = 0
        self.total_nominal_route_len = 0
        self.total_route_len = 0

class ORCA_Env(object):
    def __init__(self):
        self.t = 0
        self.stats = Sim_Stats()
        self.n_agents = N_AGENTS
        self.agents = []
        for _ in range(N_AGENTS):
            new_ac = rand_init_agent()
            self.stats.total_nominal_route_len += point_distance(new_ac.position, new_ac.destination)
            self.stats.total_route_len += DEST_THRESHOLD
            self.agents.append(new_ac)
        self.num_nmacs = [0]
        self.num_agents_arr = [N_AGENTS]

    def n_agents_control(self):
        i = 0
        while i < len(self.agents):
            if arrival(self.agents[i]):
                del self.agents[i]
            i += 1
        n_new_agents = np.random.poisson(lam=TAKEOFF_RATE*100/3600)
        for _ in range(n_new_agents):
            new_ac = rand_init_agent()
            self.stats.total_nominal_route_len += point_distance(new_ac.position, new_ac.destination)
            self.stats.total_route_len += DEST_THRESHOLD
            self.agents.append(new_ac)
        self.n_agents = len(self.agents)
        self.num_agents_arr.append(self.n_agents)

    def step(self, new_vels):
        self.num_nmacs.append(0)
        for i, agent in enumerate(self.agents):
            # limit = 'limit'
            # Limit the linear acceleration
            linear_acc = norm(new_vels[i]) - norm(agent.velocity) # linear acc
            acc_mag = np.abs(linear_acc)
            if acc_mag > MAX_ACC:
                linear_acc = min(MAX_ACC, acc_mag) * linear_acc / acc_mag

            # Limit the turn rate
            heading = norm_angle(math.atan2(agent.velocity[1], agent.velocity[0]))
            proposed_new_heading = norm_angle(math.atan2(new_vels[i][1], new_vels[i][0]))
            dheading = norm_angle(proposed_new_heading - heading)
            dheading_mag = np.abs(dheading)
            if dheading_mag > MAX_TURN:
                dheading = norm_angle(min(MAX_TURN, dheading_mag) * dheading / dheading_mag)

            heading = norm_angle(dheading + heading)
            new_speed = norm(agent.velocity) + linear_acc * DT
            agent.velocity = np.array([cos(heading), sin(heading)]) * new_speed

            # limit = ''
            # acc = (new_vels[i] - agent.velocity) / DT
            # acc_mag = np.sqrt(np.dot(acc, acc))
            # # Limit the acceleration to maximum acc
            # if acc_mag > MAX_ACC:
            #     acc = min(MAX_ACC, acc_mag) * acc / acc_mag
            # agent.velocity += acc * DT
            

            dx = agent.velocity * DT
            self.stats.total_route_len += norm(dx)
            self.stats.total_flight_hours += DT
            agent.position += dx
            # Modify preferred velocity wrt to destination
            dest_heading = norm_angle(math.atan2(
                agent.destination[1] - agent.position[1], 
                agent.destination[0] - agent.position[0]))
            agent.pref_velocity = array([MAX_SPEED * np.cos(dest_heading), 
                                         MAX_SPEED * np.sin(dest_heading)])
        self.t += 1
        self.stats.total_NMACs = np.sum(self.num_nmacs)


def norm(x):
    return np.sqrt(np.dot(x, x))

def point_distance(a, b):
    x = a - b
    return norm(x)

def distance(a, b):
    x = a.position - b.position
    return norm(x)

def get_colliders(env, agent, agent_id):
    colliders = []
    for i, ac in enumerate(env.agents):
        if i != agent_id:
            d = distance(agent, ac)
            if d <= SENSING_RANGE:
                colliders.append(ac)
            if d <= NMAC_DIST:
                env.num_nmacs[-1] += 0.5
    return colliders


# main

env = ORCA_Env()

def simulate():
    print('Take-off Rate = ', TAKEOFF_RATE)
    print('RADIUS = ', RADIUS)
    print('MAX_ACC = ', MAX_ACC)
    for _ in range(n_frames):
        new_vels = []
        frame = None
        for i, agent in enumerate(env.agents):
            colliders = get_colliders(env, agent, i)
            new_vel, _ = orca(agent, colliders, TAU, DT)
            if new_vel != 'InfeasibleError':
                new_vels.append(new_vel)
            else:
                new_vels.append(agent.velocity)    
        env.step(new_vels)
        env.n_agents_control()
        if env.t % 500 == 0:
            print('time step =', env.t, ', n_agents = ', env.n_agents)

    for agent in env.agents:
        env.stats.total_route_len += point_distance(agent.position, agent.destination)

    print('total_NMACs = ', env.stats.total_NMACs)
    print('total_flight_hours = ', env.stats.total_flight_hours)
    print('total_nominal_route_len = ', env.stats.total_nominal_route_len)
    print('total_route_len = ', env.stats.total_route_len)
    print('NMACs/flight_hour = ', env.stats.total_NMACs / env.stats.total_flight_hours)
    print('|Actual route| / |Nominal route| = ', env.stats.total_route_len / env.stats.total_nominal_route_len)



# animation:
fig = plt.figure()
plt.axes(autoscale_on=False)
def animate(i_frame):
    new_vels = []
    plt.clf()
    frame = None
    for i, agent in enumerate(env.agents):
        colliders = get_colliders(env, agent, i)
        new_vel, _ = orca(agent, colliders, TAU, DT)
        if new_vel != 'InfeasibleError':
            new_vels.append(new_vel)
        else:
            new_vels.append(agent.velocity)

    if env.t % 50 == 0 or env.t == 1:
        print('time step =', env.t, ', n_agents = ', env.n_agents)

    env.step(new_vels)
    env.n_agents_control()

    # plot
    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=3)
    for agent in env.agents:
        x = agent.position[0]
        y = agent.position[1]
        dest_x = agent.destination[0]
        dest_y = agent.destination[1]
        if arrival(agent):
            color = 'green'
        else:
            color = 'blue'
  
        ax1 = plt.scatter(x, y, marker="o", color=color, s=6)

        # arrow_len = np.sqrt(np.dot(agent.velocity, agent.velocity)) * 7
        # heading = math.atan2(agent.velocity[1], agent.velocity[0])
        # ax1 = plt.arrow(
        #     x, y,
        #     np.cos(heading) * arrow_len,
        #     np.sin(heading) * arrow_len,
        #     width=0.6,
        #     facecolor="black")

        ax1 = plt.scatter(dest_x, dest_y, marker=",", color="magenta", s=3)
        ax1 = plt.plot([x, dest_x], [y, dest_y], linestyle="--", color="black", linewidth=0.15)

        # for i in range(env.sensor_capacity):
        #     if ac.obs[4 + i * 4] < 1:
        #         rho = ac.obs[4 + i * 4] * SENSING_RANGE
        #         phi = ac.heading + ac.obs[4 + i * 4 + 1] * pi
        #         frame = plt.plot([ac.x, ac.x + rho * np.cos(phi)], [ac.y, ac.y + rho * np.sin(phi)],
        #             linestyle="--", color="red", linewidth=0.3)

    # if env.training_mode == 'circle':
    #     th = np.linspace(-pi, pi, 30)
    #     frame = plt.plot(env.circle_radius * np.cos(th), env.circle_radius * np.sin(th), 
    #         linestyle="--", color="green", linewidth=0.4)
    #     frame = plt.xlim((-env.circle_radius * 1.2, env.circle_radius * 1.2))
    #     frame = plt.ylim((-env.circle_radius * 1.2, env.circle_radius * 1.2))
    # elif env.training_mode == 'square':
    ax1 = plt.gca().set_xlim(left=0, right=AIRSPACE_WIDTH)
    ax1 = plt.gca().set_ylim(bottom=0, top=AIRSPACE_WIDTH)
        # frame = plt.gca().axis('square')
        # frame = plt.gca().set_aspect('equal')
    # elif env.training_mode == 'annulus':
    #     th = np.linspace(-pi, pi, 30)
    #     frame = plt.plot(INNER_RADIUS * np.cos(th), INNER_RADIUS * np.sin(th), 
    #         linestyle="--", color="green", linewidth=0.4)
    #     frame = plt.plot(OUTTER_RADIUS * np.cos(th), OUTTER_RADIUS * np.sin(th), 
    #         linestyle="--", color="green", linewidth=0.4)
    #     frame = plt.xlim((-OUTTER_RADIUS * 1.2, OUTTER_RADIUS * 1.2))
    #     frame = plt.ylim((-OUTTER_RADIUS * 1.2, OUTTER_RADIUS * 1.2))

    ax1 = plt.xlabel("$x$ (m)")
    ax1 = plt.ylabel("$y$ (m)")
    print(str(int(env.num_nmacs[-2])))
    ax1 = plt.title("ORCA" + ", Take-off Rate = " + str(TAKEOFF_RATE) + " flight/km$^2$-hour", fontsize=18)
    ax1 = plt.gca().set_aspect('equal')
    # frame = plt.axis("square")

    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax2 = plt.plot(range(len(env.num_agents_arr)), env.num_agents_arr, color="red")
    ax2 = plt.xlabel("Time (sec)")
    ax2 = plt.ylabel("Number of Aircraft")
    ax2 = plt.gca().set_xlim(left=0, right=1000)
    ax2 = plt.gca().set_ylim(bottom=0, top=1000)

    ax3 = plt.subplot2grid((2, 2), (1, 1))
    ax3 = plt.plot(range(len(env.num_nmacs)-1), env.num_nmacs[0:-1], color="red")
    ax3 = plt.xlabel("Time (sec)")
    ax3 = plt.ylabel("NMAC/sec")
    ax3 = plt.gca().set_xlim(left=0, right=1000)
    ax3 = plt.gca().set_ylim(bottom=0, top=25)
    # plt.draw()
    # plt.pause(0.001)
    return (ax1, ax2, ax3)


if LIMIT:
    limit = ''
else:
    limit = 'limit'

if ANIM:
    ani = animation.FuncAnimation(fig=fig, func=animate, frames=n_frames, interval=200, blit=False)
    ani.save('ORCA_' + limit + '_TOR_' + str(TAKEOFF_RATE) + '_R_' + str(RADIUS) + '_MAXACC_' + str(MAX_ACC) + 
             '_MAXTURN_' + str(int(np.rad2deg(MAX_TURN))) + '.mp4', fps=24, extra_args=['-vcodec', 'libx264'])
else:
    simulate()




