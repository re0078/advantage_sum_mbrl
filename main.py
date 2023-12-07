## script Value Summation Model-Based Reinforcement Learning
# Author   : Mehran Raisi
# Date     : 15 September 2022

import os
import argparse
import gym
import numpy as np
import tensorflow as tf
from sac import Agent
import logging
from logging import Logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, help='rl environment', required=True)
    parser.add_argument('--instance_number', type=int, help='instance of the environment', default=1)
    parser.add_argument('--load_models', type=bool, help='loading trained environment', default=False)
    parser.add_argument('--load_epoch', type=int, help='loading trained environment epoch', default=None)
    parser.add_argument('--scoring_method', type=str,
                        help="scoring function approach (either 'value' or 'advantage' )", default='advantage')
    args = parser.parse_args()

    logger = Logger("advantage sum logger")
    logger.setLevel(logging.INFO)

    env_id = args.env_id
    instance_number = args.instance_number
    load_models = args.load_models
    load_epoch = args.load_epoch
    scoring_method = args.scoring_method
    ## Creating Directory
    cwd = os.getcwd()
    directory = f'./checkpoints/{scoring_method}/{env_id}/{str(instance_number)}'
    directory = os.path.join(cwd, directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

    newpath = os.path.join(cwd, env_id)
    newpath1 = os.path.join(newpath, str(instance_number))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    if not os.path.exists(newpath1):
        os.makedirs(newpath1)
    ## Making the agent with initial parameters
    env = gym.make(env_id)
    agent = Agent(alpha=3e-4, beta=3e-4, tau=0.002,
                  input_dims=env.observation_space.shape,
                  env=env, env_id=env_id, batch_size=256, layer1_size=256, layer2_size=256,
                  n_actions=env.action_space.shape[0], instance_number=instance_number,
                  checkpt_dir=directory, scoring_method=scoring_method)
    n_games = 50_000
    if load_models:
        agent.load_models(load_epoch, logger)

    best_episode_reward = env.reward_range[0]
    step_reward_list = []
    episode_reward_list = []
    step_number = 0
    ev = 0
    save_every = 50
    reward_eval = []
    steps = []

    start_epoch = 0
    if load_models:
        start_epoch = load_epoch + 1

    for i in range(start_epoch, n_games):
        observation = env.reset()
        done = False
        ep_length = 0
        episode_reward = 0

        while not done:
            ep_length += 1
            env.render(True)
            internal_state = env.env.sim.get_state()
            # To choose each action we have to use both observation and internal_state.
            # internal_state is then used to initialize the onboard model.
            action = agent.choose_action(observation, internal_state)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            episode_reward += reward
            observation = observation_
            step_reward_list.append(reward)
            step_number += 1
            ## Eavaluating the agent per 1000 time-steps
            if step_number % 1000 == 0:
                env1 = gym.make(env_id)
                ep_r1 = 0
                observation1 = env1.reset()
                done1 = False
                ev += 1
                while not done1:
                    # observation1 = tf.convert_to_tensor([observation1], dtype=tf.float32)
                    internal_state1 = env1.env.sim.get_state()
                    # action1 = agent.actor.predict(observation1)
                    action1 = agent.choose_action(observation1, internal_state1, Test=True)
                    observation1, reward1, done1, _ = env1.step(action1)
                    ep_r1 += reward1
                reward_eval.append(ep_r1)
                logger.warning(f'Eval {ev} : {ep_r1}')
        steps.append(step_number)
        episode_reward_list.append(episode_reward)
        average_episode_reward = np.mean(episode_reward_list[-100:])
        ## Saving agent if its performance gets better
        if average_episode_reward > best_episode_reward:
            best_episode_reward = average_episode_reward
            agent.save_models(epoch=i, logger=logger)
        elif i % save_every == 0:
            agent.save_models(epoch=i, logger=logger)

        logger.warning(f'episode {i} episode reward {episode_reward} average episode reward {average_episode_reward}')
        np.save(directory + '/s_r_' + env_id + '_' + str(instance_number) + '.npy', step_reward_list)
        np.save(directory + '/ep_r_' + env_id + '_' + str(instance_number) + '.npy', episode_reward_list)
        np.save(directory + '/t_s_' + env_id + '_' + str(instance_number) + '.npy', steps)
        np.save(directory + '/eval_' + env_id + '_' + str(instance_number) + '.npy', reward_eval)
