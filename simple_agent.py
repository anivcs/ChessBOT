import gym
import random
import gym_chess

env = gym.make('Chess-v0')
obs = env.reset()
print(env.render())

done = False
while done == False:
    action = random.choice(env.legal_moves)
    obs, reward, done, info = env.step(action)
    print(env.render(mode='unicode'))
env.close()