## script Value Summation Model-Based Reinforcement Learning
# Author   : Mehran Raisi
# Date     : 15 September 2022

import os
import numpy as np
import tensorflow as tf
import gym
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent:
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8],
            env=None, env_id = 'InvertedDoublePendulum-v2', gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
            layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2, instance_number=1, 
            checkpt_dir='tmp', scoring_method='advantage'):
        self.gamma = gamma
        self.tau = tau
        # repaly buffer to store real env transitions
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        # replay buffer to store onboard model transitions and trainging the crtitc networks
        self.memory_critic_only = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        
        self.env_id = env_id
        # Number of trajectories to be generated
        self.paths_per_cpu = 4
        # Number of horizons per each Trajectory
        self.plan_horizon = 4
        self.gamma_mppi = gamma
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.instance_number = instance_number
        chkpt_dir = checkpt_dir
        self.chkpt_dir = chkpt_dir
        self.scoring_method = scoring_method

        self.actor = ActorNetwork(n_actions=n_actions, name='actor', 
                                    max_action=env.action_space.high, chkpt_dir=chkpt_dir)
        self.critic_1 = CriticNetwork(n_actions=n_actions, name='critic_1', chkpt_dir=chkpt_dir)
        self.critic_2 = CriticNetwork(n_actions=n_actions, name='critic_2', chkpt_dir=chkpt_dir)
        self.value = ValueNetwork(name='value', chkpt_dir=chkpt_dir)
        self.target_value = ValueNetwork(name='target_value', chkpt_dir=chkpt_dir)

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.critic_2.compile(optimizer=Adam(learning_rate=beta))
        self.value.compile(optimizer=Adam(learning_rate=beta))
        self.target_value.compile(optimizer=Adam(learning_rate=beta))

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def remember(self, state, action, reward, new_state, done):
        # Storing real env transitions
        self.memory.store_transition(state, action, reward, new_state, done)

    def remember_critic(self, state, action, reward, new_state, done):
        # Saving onboard model transitions
        self.memory_critic_only.store_transition(state, action, reward, new_state, done)

    def learn(self):
        # Here we update the parameters of all networks using SAC algorithm
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            value = tf.squeeze(self.value(states), 1)
            value_ = tf.squeeze(self.target_value(states_), 1)

            current_policy_actions, log_probs = self.actor.sample_normal(states,
                                                        reparameterize=False)
            log_probs = tf.squeeze(log_probs,1)
            q1_new_policy = self.critic_1(states, current_policy_actions)
            q2_new_policy = self.critic_2(states, current_policy_actions)
            critic_value = tf.squeeze(
                                tf.math.minimum(q1_new_policy, q2_new_policy), 1)


            value_target = critic_value - log_probs
            value_loss = 0.5 * keras.losses.MSE(value, value_target)

        value_network_gradient = tape.gradient(value_loss, 
                                                self.value.trainable_variables)
        self.value.optimizer.apply_gradients(zip(
                       value_network_gradient, self.value.trainable_variables))


        with tf.GradientTape() as tape:
            new_policy_actions, log_probs = self.actor.sample_normal(states,
                                                reparameterize=True)
            log_probs = tf.squeeze(log_probs, 1)
            q1_new_policy = self.critic_1(states, new_policy_actions)
            q2_new_policy = self.critic_2(states, new_policy_actions)
            critic_value = tf.squeeze(tf.math.minimum(
                                        q1_new_policy, q2_new_policy), 1)
        
            actor_loss = log_probs - critic_value
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, 
                                            self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
                        actor_network_gradient, self.actor.trainable_variables))
        

        with tf.GradientTape(persistent=True) as tape:
            # I didn't know that these context managers shared values?
            q_hat = self.scale*reward + self.gamma*value_*(1-done)
            q1_old_policy = tf.squeeze(self.critic_1(state, action), 1)
            q2_old_policy = tf.squeeze(self.critic_2(state, action), 1)
            critic_1_loss = 0.5 * keras.losses.MSE(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * keras.losses.MSE(q2_old_policy, q_hat)
    
        critic_1_network_gradient = tape.gradient(critic_1_loss,
                                        self.critic_1.trainable_variables)
        critic_2_network_gradient = tape.gradient(critic_2_loss,
            self.critic_2.trainable_variables)

        self.critic_1.optimizer.apply_gradients(zip(
            critic_1_network_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(
            critic_2_network_gradient, self.critic_2.trainable_variables))

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_value.weights
        for i, weight in enumerate(self.value.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_value.set_weights(weights)

    def learn_critic(self):
        # Here we update the params of value and critic networks only using data sampled from
        # onboard model replay buffer
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory_critic_only.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            value = tf.squeeze(self.value(states), 1)
            value_ = tf.squeeze(self.target_value(states_), 1)

            current_policy_actions, log_probs = self.actor.sample_normal(states,
                                                        reparameterize=False)
            log_probs = tf.squeeze(log_probs,1)
            q1_new_policy = self.critic_1(states, current_policy_actions)
            q2_new_policy = self.critic_2(states, current_policy_actions)
            critic_value = tf.squeeze(
                                tf.math.minimum(q1_new_policy, q2_new_policy), 1)

            value_target = critic_value - log_probs
            value_loss = 0.5 * keras.losses.MSE(value, value_target)


        value_network_gradient = tape.gradient(value_loss, 
                                                self.value.trainable_variables)
        self.value.optimizer.apply_gradients(zip(
                       value_network_gradient, self.value.trainable_variables))

        with tf.GradientTape(persistent=True) as tape:
            # I didn't know that these context managers shared values?
            q_hat = self.scale*reward + self.gamma*value_*(1-done)
            q1_old_policy = tf.squeeze(self.critic_1(state, action), 1)
            q2_old_policy = tf.squeeze(self.critic_2(state, action), 1)
            critic_1_loss = 0.5 * keras.losses.MSE(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * keras.losses.MSE(q2_old_policy, q_hat)
    
        critic_1_network_gradient = tape.gradient(critic_1_loss,
                                        self.critic_1.trainable_variables)
        critic_2_network_gradient = tape.gradient(critic_2_loss,
            self.critic_2.trainable_variables)

        self.critic_1.optimizer.apply_gradients(zip(
            critic_1_network_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(
            critic_2_network_gradient, self.critic_2.trainable_variables))

        self.update_value_network_parameters()

    def update_value_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_value.weights
        for i, weight in enumerate(self.value.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_value.set_weights(weights)

    def save_models(self, epoch):
        print('... saving models on epoch {epoch} ...')
        saving_dir = os.path.join(self.chkpt_dir, str(epoch))
        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)
        self.actor.save_weights(os.path.join(self.actor.checkpoint_file, str(epoch)))
        self.critic_1.save_weights(os.path.join(self.critic_1.checkpoint_file, str(epoch)))
        self.critic_2.save_weights(os.path.join(self.critic_2.checkpoint_file, str(epoch)))
        self.value.save_weights(os.path.join(self.value.checkpoint_file, str(epoch)))
        self.target_value.save_weights(os.path.join(self.target_value.checkpoint_file, str(epoch)))
        # print('does path exist? ', f'{self.chkpt_dir}/{epoch}/memory_rp.npz')
        self.memory.save_to_file(f'{self.chkpt_dir}/{epoch}/memory_rp.npz')
        self.memory_critic_only.save_to_file(f'{self.chkpt_dir}/{epoch}/env_critic_only_rb.npz')

    def load_models(self, epoch):
        print('... loading models ...')
        self.actor.load_weights(os.path.join(self.actor.checkpoint_file, str(epoch)))
        self.critic_1.load_weights(os.path.join(self.critic_1.checkpoint_file, str(epoch)))
        self.critic_2.load_weights(os.path.join(self.critic_2.checkpoint_file, str(epoch)))
        self.value.load_weights(os.path.join(self.value.checkpoint_file, str(epoch)))
        self.target_value.load_weights(os.path.join(self.target_value.checkpoint_file, str(epoch)))
        self.memory.load_from_file(f'{self.chkpt_dir}/{epoch}/memory_rp.npz')
        self.memory_critic_only.load_from_file(f'{self.chkpt_dir}/{epoch}/env_critic_only_rb.npz')


    def choose_action(self, observation, internal_state, Test=False):
        # Here N trajectory is gathered using onboard model and among them the one
        # with the highest score is chosen to be applied tho the real world environment
        paths, act_list = self.do_env_rollout(observation, internal_state, Test)
        R = self.score_trajectory(paths)
        action = act_list[np.argmax(R)]
        mu_prime = action[0].copy()
        return mu_prime

    def do_env_rollout(self, observation, internal_state, Test=False):
        e = gym.make(self.env_id)
        paths = []
        act_list = []
        for i in range(self.paths_per_cpu):
            e.reset()
            e.env.sim.set_state(internal_state)
            obs = []
            rewards = []
            values = []
            act_sequence = []
            for k in range(self.plan_horizon):
                obs.append(self.get_obs(e))
                state = tf.convert_to_tensor([observation], dtype=tf.float32)
                mu, _ = self.actor.sample_normal(state, reparameterize=False)
                act_sequence.append(mu[0])
                s, r, d, ifo = e.step(mu[0])
                st = tf.convert_to_tensor([obs[-1]], dtype=tf.float32)
                v1 = self.critic_1(st, mu).numpy()[0][0]
                v2 = self.critic_2(st, mu).numpy()[0][0]
                v = min(v1, v2)

                if self.scoring_method == "advantage":
                    statev = self.value(st).numpy()[0][0]
                    a = v - statev
                    values.append(a)
                elif self.scoring_method == "value":
                    values.append(v)
                else:
                    raise NotImplementedError

                if not Test:
                    self.remember_critic(obs[-1], act_sequence[-1], r, s, d)
                rewards.append(r)
            path = dict(observations=np.array(obs),
                        actions=np.array(act_sequence),
                        rewards=np.array(rewards),
                        values = np.array(values),)
            paths.append(path)
            act_sequence = np.array(act_sequence)
            act_list.append(act_sequence)
            if not Test :
                self.learn_critic()
        del(e)
        return paths, act_list

    def score_trajectory(self, paths):
        scores = np.zeros(len(paths))
        for i in range(len(paths)):
            scores[i] = 0.0
            for t in range(paths[i]["values"].shape[0]):
                scores[i] += (self.gamma_mppi**t)*paths[i]["values"][t]
        return scores

    def get_obs(self, env):
        if self.env_id == 'Walker2d-v3':
            position = env.env.env.sim.data.qpos.flat.copy()
            velocity = np.clip(env.env.env.sim.data.qvel.flat.copy(), -10, 10)
            position = position[1:]
            observation = np.concatenate((position, velocity)).ravel()
            return observation

        elif self.env_id == 'Hopper-v3':
            position = env.env.env.sim.data.qpos.flat.copy()
            velocity = np.clip(env.env.env.sim.data.qvel.flat.copy(), -10, 10)
            position = position[1:]
            observation = np.concatenate((position, velocity)).ravel()
            return observation
        
        elif self.env_id == 'InvertedDoublePendulum-v2' :
            return np.concatenate([ env.env.env.sim.data.qpos[:1],
                                    np.sin(env.env.env.sim.data.qpos[1:]),
                                    np.cos(env.env.env.sim.data.qpos[1:]),
                                    np.clip(env.env.env.sim.data.qvel, -10, 10),
                                    np.clip(env.env.env.sim.data.qfrc_constraint, -10, 10),]).ravel()

        elif self.env_id == 'HalfCheetah-v3':
            position = env.env.env.sim.data.qpos.flat.copy()
            velocity = env.env.env.sim.data.qvel.flat.copy()
            position = position[1:]
            observation = np.concatenate((position, velocity)).ravel()
            return observation

        elif self.env_id == 'Ant-v3' :
            position = env.env.env.sim.data.qpos.flat.copy()
            velocity = env.env.env.sim.data.qvel.flat.copy()
            contact_force = env.env.env.contact_forces.flat.copy()
    
            position = position[2:]
    
            observations = np.concatenate((position, velocity, contact_force))
    
            return observations

        elif self.env_id == 'Humanoid-v3' :
            position = env.env.env.sim.data.qpos.flat.copy()
            velocity = env.env.env.sim.data.qvel.flat.copy()
    
            com_inertia = env.env.env.sim.data.cinert.flat.copy()
            com_velocity = env.env.env.sim.data.cvel.flat.copy()
    
            actuator_forces = env.env.env.sim.data.qfrc_actuator.flat.copy()
            external_contact_forces = env.env.env.sim.data.cfrc_ext.flat.copy()
    
            position = position[2:]
    
            return np.concatenate((
                                   position,
                                   velocity,
                                   com_inertia,
                                   com_velocity,
                                   actuator_forces,
                                   external_contact_forces,))

        elif self.env_id == 'HumanoidStandup-v2' :
            data = env.env.env.sim.data
            return np.concatenate(
                [
                    data.qpos.flat[2:],
                    data.qvel.flat,
                    data.cinert.flat,
                    data.cvel.flat,
                    data.qfrc_actuator.flat,
                    data.cfrc_ext.flat,
                ])


        elif self.env_id == 'Pusher-v2' :
            data = env.env.sim.data
            return np.concatenate([
                data.qpos.flat[:7],
                data.qvel.flat[:7],
                env.env.get_body_com("tips_arm"),
                env.env.get_body_com("object"),
                env.env.get_body_com("goal"),
                ])
        elif self.env_id == 'Reacher-v2' :
            theta = env.env.env.sim.data.qpos.flat[:2]
            return np.concatenate([
                np.cos(theta),
                np.sin(theta),
                env.env.env.sim.data.qpos.flat[2:],
                env.env.env.sim.data.qvel.flat[:2],
                env.env.env.get_body_com("fingertip") - env.env.env.get_body_com("target"),
                ])
        else :
            print(self.env_id, ' is not found')
