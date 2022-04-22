import gym
from gym.wrappers import TransformReward
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.buffers import ReplayBuffer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed",
                    required=True,
                    default=0,
                    type=int)
args, unknown = parser.parse_known_args()
print("Training for Seed={}".format(args.seed))
train_env = gym.make("BipedalWalker-v3")
train_env = TransformReward(train_env, lambda x: np.clip(x, a_min=-20, a_max=20))
test_env = gym.make("BipedalWalker-v3")
policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=dict(pi=[256, 256], qf=[256, 256]))
noise = NormalActionNoise(mean=0, sigma=0.1)
model = SAC("MlpPolicy", train_env, learning_rate=0.0003, buffer_size=200000, learning_starts=10000, \
            batch_size=512, tau=0.005, action_noise=noise, replay_buffer_class=ReplayBuffer, \
            ent_coef=0.1, seed=args.seed, verbose=1, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=1100000, eval_env=test_env, eval_freq=10000, n_eval_episodes=5)
model.save("SAC_BipedalWalker")