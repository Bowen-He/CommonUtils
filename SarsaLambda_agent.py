import numpy as np
from collections import deque
from .base import base


class SarsaLambda_agent(base):
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
            lambda_value=0.9,
            initialization=0
    ):
        super(SarsaLambda_agent, self).__init__(
            env,
            num_states,
            num_actions,
            alpha,
            gamma,
            start_epsilon,
            end_epsilon,
            exploration_fraction,
        )
        self.lambda_value = lambda_value
        self.qtable = np.ones((self.num_states, self.num_actions))
        self.qtable = self.qtable * initialization
        self.etable = np.zeros((self.num_states, self.num_actions))

    def set_lambda(self, lambda_value):
        self.lambda_value = lambda_value

    def write(self):
        np.save("qtable", self.qtable)
        np.save("etable", self.etable)

    def read(self, qtable_name=None, etable_name=None):
        qtable = np.load(qtable_name)
        etable = np.load(etable_name)
        assert qtable.shape[0] == self.num_states and qtable.shape[1] == self.num_actions
        assert etable.shape[0] == self.num_states and etable.shape[1] == self.num_actions

        self.qtable = qtable
        self.etable = etable

    def learn(self, num_episodes, log_dir=None, verbose=False, log_frequency=100):
        self.epsilon = self.start_epsilon
        self.epsilon_decay_rate = (self.start_epsilon - self.end_epsilon) / (self.exploration_fraction*num_episodes)
        step_queue, reward_queue = deque(maxlen=log_frequency), deque(maxlen=log_frequency)
        step_list, reward_list, table_list = [], [], []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            action = self.sample_action(state)
            self.etable = np.zeros((self.num_states, self.num_actions))
            done = False
            episode_reward = 0
            episode_step = 0
            while not done:
                EnvStep = self.env.step(action)
                next_state, reward, done = EnvStep.observation, EnvStep.reward, EnvStep.last
                next_action = self.sample_action(next_state)
                delta = reward + self.gamma * (1 - int(done)) * self.qtable[next_state, next_action] - self.qtable[state, action]
                self.etable[state, action] += 1
                if not delta == 0:
                    self.qtable = self.qtable + self.alpha*delta*self.etable
                    self.etable = self.gamma * self.lambda_value*self.etable
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