import gym
import numpy as np
from dueling_ddqn_torch import Agent
from utils import plotLearning

if __name__ == '__main__':
    env = gym.make('LunarLander-v2', render_mode="human")

    done = False
    observation = env.reset()
    score = 0

    while not done:
        if len(observation) == 2:
            observation = observation[0]
        env.render()
        action = env.action_space.sample()
        observation_, reward, done, _, info = env.step(action)
        score += reward
        observation = observation_