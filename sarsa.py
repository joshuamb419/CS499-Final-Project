import pickle

import minigrid as mg
import gymnasium as gym
import numpy as np
import random
from gymnasium.wrappers import FlattenObservation
from gymnasium.spaces.utils import flatten_space

def choose_action(state, Q):
    if random.random() < epsilon:
        return random.randint(0, 3)
    else:
        return np.argmax(Q[state][:])

def improve(s1, a1, r, s2, a2, Q):
    original = Q[s1][a1]
    target = r + gamma * Q[s2][a2]
    Q[s1][a1] = Q[s1][a1] + learn_rate * (target - original)

def check_state(s, Q):
    if s not in Q:
        Q[s] = np.zeros((4,1))

def train(env):
    Q = {}
    episode = 0
    while episode < max_episodes:
        observation = env.reset()
        step = 0
        done = False
        # print(observation)

        s = (observation[0]['direction'], observation[0]['image'].data.tobytes())
        check_state(s, Q)
        a = choose_action(s, Q)

        while step < max_steps:
            # print(env.step(a))
            o2, r, done, info, info2 = env.step(a)
            s2 = (o2['direction'], o2['image'].data.tobytes())
            check_state(s2, Q)
            a2 = choose_action(s2, Q)

            improve(s, a, r, s2, a2, Q)

            s = s2
            a = a2
            step += 1
            if done:
                break

        episode += 1
        print(f'episode {episode}, Done: {done}, Steps: {step}')
    return Q

env = gym.make("MiniGrid-FourRooms-v0")

epsilon = 0.3
gamma = 0.95
learn_rate = 0.8
max_episodes = 1000
max_steps = 10000

trained_Q = train(env)
with open('sarsa_q.dict', 'wb') as file:
    pickle.dump(trained_Q, file)
# print(trained_Q)

