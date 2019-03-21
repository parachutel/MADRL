import sys
sys.path.append('./rllab')
sys.path.append('./rltools')
import os

import argparse
import joblib
import tensorflow as tf
import numpy as np

from madrl_environments.cas.multi_aircraft import *

from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["figure.figsize"] = [5, 5]


def main():
    n_frames = 350
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_agents', type=int, default=15)
    parser.add_argument('--equally_spaced_circle', type=int, default=0) # overrides --mode

    parser.add_argument('--mode', type=str, default='circle')
    parser.add_argument('--policy', type=str, default='trpo_full_curr_3_passes_ALL_ENV_Tmax_300_PEN_HEAVY_True_rew_nmac_-150_rew_arr_2_run_2/itr_299.pkl')
    args = parser.parse_args()

    args.equally_spaced_circle = bool(args.equally_spaced_circle)

    print('Animating policy: {}'.format(args.policy))
    if args.equally_spaced_circle:
        n_frames = 500
        args.mode = 'circle'
        print('Using equally_spaced_circle, --mode overridden')
    print('Scenario: {}'.format(args.mode))
    print('Number of agents = {}'.format(args.n_agents))

    tf.reset_default_graph()
    with tf.Session() as sess:
        # policy:
        data = joblib.load('./rllab/data/' + args.policy)
        policy = data['policy']

        # env:
        env = MultiAircraftEnv(n_agents=args.n_agents, random_mode=False, training_mode=args.mode)
        if args.equally_spaced_circle:
            # Manually reset:
            env.reset()
            args.mode = 'equally_spaced_circle'
            env.training_mode = 'circle'
            CIRCLE_RADIUS = 6000
            env.circle_radius = CIRCLE_RADIUS
            thetas = np.linspace(0, 2 * pi, args.n_agents + 1)
            for i in range(args.n_agents):
                env.aircraft[i].x = CIRCLE_RADIUS * np.cos(thetas[i])
                env.aircraft[i].y = CIRCLE_RADIUS * np.sin(thetas[i])
                env.aircraft[i].dest_x = CIRCLE_RADIUS * np.cos(thetas[i] + pi)
                env.aircraft[i].dest_y = CIRCLE_RADIUS * np.sin(thetas[i] + pi)
                env.aircraft[i].heading = thetas[i] + pi
                env.aircraft[i].dist_to_dest = 2 * CIRCLE_RADIUS
                env.aircraft[i].init_dist_to_dest = env.aircraft[i].dist_to_dest
                env.aircraft[i].prev_dist_to_dest = env.aircraft[i].dist_to_dest
                env.aircraft[i].v = 30
                env.aircraft[i].turn_rate = 0
        else:
            env.reset()

        # animation:
        fig = plt.figure()
        plt.axes(autoscale_on=False)
        def animate(i):
            actions = []
            plt.clf()
            frame = None
            for ac in env.aircraft:
                obs = ac.get_observation()
                _, action_info = policy.get_action(obs)
                actions.append(action_info['mean'])

            if env.t % 50 == 0 or env.t == 1:
                print('time step =', env.t)
            env.step(np.array(actions))

            # plot
            for ac in env.aircraft:
                if ac.arrival():
                    color = 'green'
                else:
                    color = 'blue'
                frame = plt.scatter(ac.x, ac.y, marker="o", color=color, s=12)
                arrow_len = ac.v * 10
                frame = plt.arrow(
                    ac.x, ac.y,
                    np.cos(ac.heading) * arrow_len,
                    np.sin(ac.heading) * arrow_len,
                    width=0.6,
                    facecolor="black")
                frame = plt.scatter(ac.dest_x, ac.dest_y, marker=",", color="magenta", s=12)
                frame = plt.plot([ac.x, ac.dest_x], [ac.y, ac.dest_y], 
                    linestyle="--", color="black", linewidth=0.3)
                for i in range(env.sensor_capacity):
                    if ac.obs[4 + i * 4] < 1:
                        rho = ac.obs[4 + i * 4] * SENSING_RANGE
                        phi = ac.heading + ac.obs[4 + i * 4 + 1] * pi
                        frame = plt.plot([ac.x, ac.x + rho * np.cos(phi)], [ac.y, ac.y + rho * np.sin(phi)],
                            linestyle="--", color="red", linewidth=0.3)
        
            if env.training_mode == 'circle':
                th = np.linspace(-pi, pi, 30)
                frame = plt.plot(env.circle_radius * np.cos(th), env.circle_radius * np.sin(th), 
                    linestyle="--", color="green", linewidth=0.4)
                frame = plt.xlim((-env.circle_radius * 1.2, env.circle_radius * 1.2))
                frame = plt.ylim((-env.circle_radius * 1.2, env.circle_radius * 1.2))
            elif env.training_mode == 'square':
                frame = plt.gca().set_xlim(left=0, right=AIRSPACE_WIDTH)
                frame = plt.gca().set_ylim(bottom=0, top=AIRSPACE_WIDTH)
                frame = plt.gca().axis('equal')
            elif env.training_mode == 'annulus':
                th = np.linspace(-pi, pi, 30)
                frame = plt.plot(INNER_RADIUS * np.cos(th), INNER_RADIUS * np.sin(th), 
                    linestyle="--", color="green", linewidth=0.4)
                frame = plt.plot(OUTTER_RADIUS * np.cos(th), OUTTER_RADIUS * np.sin(th), 
                    linestyle="--", color="green", linewidth=0.4)
                frame = plt.xlim((-OUTTER_RADIUS * 1.2, OUTTER_RADIUS * 1.2))
                frame = plt.ylim((-OUTTER_RADIUS * 1.2, OUTTER_RADIUS * 1.2))

            frame = plt.xlabel("x (m)")
            frame = plt.ylabel("y (m)")
            frame = plt.title('t = ' + str(env.t) + ', Num Agents = ' + str(len(env.aircraft)))
            frame = plt.axis("equal")

            return frame
        
        ani = animation.FuncAnimation(fig=fig, func=animate, frames=n_frames, interval=200, blit=False)
        ani.save('animation_' + str(env.n_agents) + '_' + args.mode + '_' + os.path.split(args.policy)[0] + '.mp4', fps=24, extra_args=['-vcodec', 'libx264'])

if __name__ == '__main__':
    main()

