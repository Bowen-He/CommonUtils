import numpy as np
from .base import base, sample_action_with_fixed_policy
from collections import deque


class Sarsa_agent(base):
    def __init__(
            self,
            env,
            num_states=500,
            num_actions=6,
            alpha=0.2,
            gamma=0.9,
            start_epsilon=1,
            end_epsilon=0.01,
            exploration_fraction=0.1,
            initialization=0
    ):
        super(Sarsa_agent, self).__init__(
            env,
            num_states,
            num_actions,
            alpha,
            gamma,
            start_epsilon,
            end_epsilon,
            exploration_fraction,
        )
        self.qtable = np.ones((self.num_states, self.num_actions))
        self.qtable = self.qtable*initialization

    def learn(self, num_episodes, log_dir=None, verbose=False, log_frequency=100):
        self.epsilon = self.start_epsilon
        self.epsilon_decay_rate = (self.start_epsilon - self.end_epsilon) / (self.exploration_fraction*num_episodes)
        step_queue, reward_queue = deque(maxlen=log_frequency), deque(maxlen=log_frequency)
        step_list, reward_list, table_list = [], [], []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            action = self.sample_action(state)
            done = False
            episode_reward = 0
            episode_step = 0
            while not done:
                EnvStep = self.env.step(action)
                next_state, reward, done = EnvStep.observation, EnvStep.reward, EnvStep.last
                next_action = self.sample_action(next_state)
                self.qtable[state, action] = (1 - self.alpha) * self.qtable[state, action] + \
                                             self.alpha * (reward + self.gamma * (1 - int(done)) * self.qtable[next_state, next_action])
                episode_reward += reward
                episode_step += 1
                state = next_state
                action = next_action
            step_queue.append(episode_step)
            reward_queue.append(episode_reward)

            if episode % log_frequency == 0:
                step_list.append(np.asscalar(np.average(step_queue)))
                reward_list.append(np.asscalar(np.average(reward_queue)))
                table_list.append(self.qtable[0, :].tolist())
                np.save(log_dir + "/QTable", np.array(table_list))
                np.save(log_dir + "/Step", np.array(step_list))
                np.save(log_dir + "/Reward", np.array(reward_list))

            if verbose:
                print("episode {:g} reward:{:g} steps:{:g} epsilon:{:g}".format(episode, episode_reward, episode_step, self.epsilon))

            if self.epsilon > self.end_epsilon:
                self.epsilon = self.epsilon - self.epsilon_decay_rate

    def experiment_learn(self, num_episodes):
        self.epsilon = self.start_epsilon
        self.epsilon_decay_rate = (self.start_epsilon - self.end_epsilon) / (self.exploration_fraction * num_episodes)

        for episode in range(num_episodes):
            state, reward, done, _ = self.env.reset()
            action = self.sample_action(state)
            episode_reward = 0
            episode_step = 0
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                next_action = self.sample_action(next_state)
                self.qtable[state, action] = (1 - self.alpha) * self.qtable[state, action] + \
                                             self.alpha * (reward + self.gamma * (1 - int(done)) * self.qtable[next_state, next_action])
                episode_reward += reward
                episode_step += 1
                state = next_state
                action = next_action

            if self.epsilon > self.end_epsilon:
                self.epsilon = self.epsilon - self.epsilon_decay_rate


