import gym
import random
import gym_chess
import numpy as np
import torch as th
import matplotlib.pyplot as plt 

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('Chess-v0')

tensorboard_log = "data/tb/"

dqn_model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    train_freq=16,
    gradient_steps=8,
    gamma=0.99,
    exploration_fraction=0.2,
    exploration_final_eps=0.07,
    target_update_interval=600,
    learning_starts=1000,
    buffer_size=10000,
    batch_size=128,
    learning_rate=4e-3,
    policy_kwargs=dict(net_arch=[256, 256]),
    tensorboard_log=tensorboard_log,
    seed=2,
)

obs = env.reset()
obs_space = env.observation_space
action_space = env.action_space
print("The observation space: {}".format(obs_space))
print("The action space: {}".format(action_space))
print(env.render())
print("The initial observation is {}".format(obs))



done = False
while done == False:
    action = random.choice(env.legal_moves)
    obs, reward, done, info = env.step(action)
    print(env.render(mode='unicode'))
    print("The new observation is {}".format(obs))
env.close()
