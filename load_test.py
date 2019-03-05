import joblib
import tensorflow as tf
import numpy as np
from rllab.misc import tensor_utils
import time

data_file_path = './rllab/data/experiment_2019_03_05_09_54_11_296673_PST_08c0f/itr_4.pkl'

def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated:
        env.render(close=True)

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )


with tf.Session() as sess:
    data = joblib.load(data_file_path)
    policy = data['policy']
    env = data['env']
    print(env.max_time_steps)
    # path = rollout(env, policy, max_path_length=1000, 
    #                 animated=False, speedup=1)


# run_experiment