import sys
sys.path.append('../../../rllab')
sys.path.append('../../../rltools')
sys.path.append('../../../madrl_environments')

from madrl_environments.cas.multi_aircraft import *

import argparse
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["figure.figsize"] = [3, 3]

import joblib
import tensorflow as tf
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='')
    parser.add_argument('--n_agents', type=int, default=2)
    parser.add_argument('--n_eval', type=int, default=1)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--time_steps', type=int, default=300)
    parser.add_argument('--equally_spaced_circle', type=bool, default=False)
    parser.add_argument('--ylabel', type=bool, default=True)
    args = parser.parse_args()

    mean_extra_traj_len = []
    avg_speed = []

    print('Evaluating policy: {}'.format(args.policy))
    print('Number of agents = {}'.format(args.n_agents))

    policy_file = '../../../rllab/data/' + args.policy
    tf.reset_default_graph()
    with tf.Session() as sess:
        if args.policy != 'none':
            policy_file_data = joblib.load(policy_file)
            policy = policy_file_data['policy']

        for _ in range(args.n_eval):
            env = MultiAircraftEnv(n_agents=args.n_agents, 
                                    render_option=False,
                                    constant_n_agents=True,
                                    random_mode=True)
            env.reset()
            
            # if args.equally_spaced_circle:
            #     # Manually reset:
            #     env.training_mode = 'circle'
            #     CIRCLE_RADIUS = 6000
            #     env.circle_radius = CIRCLE_RADIUS
            #     thetas = np.linspace(0, 2 * pi, args.n_agents + 1)
            #     for i in range(args.n_agents):
            #         env.aircraft[i].x = CIRCLE_RADIUS * np.cos(thetas[i])
            #         env.aircraft[i].y = CIRCLE_RADIUS * np.sin(thetas[i])
            #         env.aircraft[i].dest_x = CIRCLE_RADIUS * np.cos(thetas[i] + pi)
            #         env.aircraft[i].dest_y = CIRCLE_RADIUS * np.sin(thetas[i] + pi)
            #         env.aircraft[i].heading = thetas[i] + pi
            #         env.aircraft[i].dist_to_dest = 2 * CIRCLE_RADIUS
            #         env.aircraft[i].init_dist_to_dest = env.aircraft[i].dist_to_dest
            #         env.aircraft[i].prev_dist_to_dest = env.aircraft[i].dist_to_dest
            #         env.aircraft[i].v = 30
            #         env.aircraft[i].turn_rate = 0
    
            traj = {}
            for i in range(args.n_agents):
                traj[i] = {}
                traj[i]['x'] = [env.aircraft[i].x]
                traj[i]['y'] = [env.aircraft[i].y]   
                traj[i]['extra_len'] = 0
                traj[i]['avg_speed'] = [env.aircraft[i].v]
                traj[i]['travel_time'] = 0
                
            t = 0
            while t < args.time_steps:
                actions = []
                for ac in env.aircraft:
                    obs = ac.get_observation()
                    if args.policy != 'none':
                        _, action_info = policy.get_action(obs)
                        actions.append(action_info['mean'])
                    else:
                        actions.append([0] * ACTION_DIM)
                
                env.step(np.array(actions))
                t += 1
                for i in range(args.n_agents):
                    traj[i]['x'].append(env.aircraft[i].x)
                    traj[i]['y'].append(env.aircraft[i].y)
                    if not env.aircraft[i].arrival():
                        traj[i]['avg_speed'].append(env.aircraft[i].v)
                        traj[i]['extra_len'] += np.sqrt((traj[i]['x'][t] - traj[i]['x'][t - 1])**2 + (traj[i]['y'][t] - traj[i]['y'][t - 1])**2)
                    if traj[i]['travel_time'] == 0 and env.aircraft[i].arrival():
                        traj[i]['travel_time'] = t
    
                    # if t == 1:
                    #     plt.arrow(
                    #         env.aircraft[i].x, env.aircraft[i].y,
                    #         np.cos(env.aircraft[i].heading) * 100,
                    #         np.sin(env.aircraft[i].heading) * 100,
                    #         width=50,
                    #         facecolor="black")
    
                
    
            for i in range(args.n_agents):
                traj[i]['avg_speed'] = np.mean(np.array(traj[i]['avg_speed'])) / MAX_V
                traj[i]['extra_len'] /= env.aircraft[i].init_dist_to_dest
                traj[i]['extra_len'] -= 1
                avg_speed.append(traj[i]['avg_speed'])
                mean_extra_traj_len.append(traj[i]['extra_len'])
    
                # plt.plot(traj[i]['x'], traj[i]['y'])
                # plt.scatter(traj[i]['x'][0], traj[i]['y'][0], marker="o", color='blue', s=20)
                # plt.scatter(env.aircraft[i].dest_x, env.aircraft[i].dest_y, marker=",", color="magenta", s=20)

        ste_extra_traj_len = np.std(mean_extra_traj_len) / np.sqrt(args.n_eval)
        mean_extra_traj_len = np.mean(mean_extra_traj_len)
        ste_avg_speed = np.std(avg_speed) / np.sqrt(args.n_eval)
        mean_avg_speed = np.mean(avg_speed)

        print('avg_speed = %3.2f $\\pm$ %3.2f' % (mean_avg_speed, ste_avg_speed))
        print('mean_extra_traj_len = %3.2f $\\pm$ %3.2f' % (mean_extra_traj_len, ste_extra_traj_len))

        # plt.axis("equal")
        # plt.title(env.training_mode + ', Number of Agents: ' + str(len(env.aircraft)))
        # plt.xlabel("x (m)")
        # if args.ylabel:
        #     plt.ylabel("y (m)")
        # else:
        #     plt.gca().axes.get_yaxis().set_ticks([])

        # if env.training_mode == 'circle':
        #     th = np.linspace(-pi, pi, 30)
        #     plt.plot(env.circle_radius * np.cos(th), env.circle_radius * np.sin(th), 
        #         linestyle="--", color="green", linewidth=0.4)
        #     plt.xlim((-env.circle_radius * 1.2, env.circle_radius * 1.2))
        #     plt.ylim((-env.circle_radius * 1.2, env.circle_radius * 1.2))
        # elif env.training_mode == 'square':
        #     plt.gca().set_xlim(left=0, right=AIRSPACE_WIDTH)
        #     plt.gca().set_ylim(bottom=0, top=AIRSPACE_WIDTH)
        #     plt.gca().axis('equal')
        # elif env.training_mode == 'annulus':
        #     th = np.linspace(-pi, pi, 30)
        #     plt.plot(INNER_RADIUS * np.cos(th), INNER_RADIUS * np.sin(th), 
        #         linestyle="--", color="green", linewidth=0.4)
        #     plt.plot(OUTTER_RADIUS * np.cos(th), OUTTER_RADIUS * np.sin(th), 
        #         linestyle="--", color="green", linewidth=0.4)
        #     plt.xlim((-OUTTER_RADIUS * 1.2, OUTTER_RADIUS * 1.2))
        #     plt.ylim((-OUTTER_RADIUS * 1.2, OUTTER_RADIUS * 1.2))

        # plt.savefig('traj_vis_' + env.training_mode + '_n_' + str(len(env.aircraft)) + '_.pdf', bbox_inches='tight', pad_inches=0)
        # plt.show()
        

    
if __name__ == '__main__':
    main()