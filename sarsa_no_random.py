import pickle
import sys

import minigrid as mg
import gymnasium as gym
import numpy as np
import random
import csv

from gymnasium.wrappers import TimeLimit
from minigrid.wrappers import SymbolicObsWrapper

def choose_action(state, Q):
    if random.random() < epsilon:
        return random.randint(0, 2)
    else:
        if Q[state][0] == Q[state][1] and Q[state][1] == Q[state][2]:
            return 2
        else:
            return np.argmax(Q[state][:])

def improve(s1, a1, r, s2, a2, Q):
    original = Q[s1][a1]
    target = r + gamma * Q[s2][a2]
    Q[s1][a1] = Q[s1][a1] + learn_rate * (target - original)

def check_state(s, Q):
    if s not in Q:
        Q[s] = np.zeros((3,1))

def train(env, Q = {}):
    episode = 0
    steps_arr = []
    reward_arr = []
    while episode < max_episodes:
        observation = env.reset(seed=env.np_random_seed)

        step = 0
        total_reward = 0
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
            total_reward += r
            if done:
                break

        episode += 1
        steps_arr.append(step)
        reward_arr.append(total_reward)
        print(f'episode {episode}, Done: {done}, Steps: {step}, Reward: {total_reward}')
    return (Q, steps_arr, reward_arr)


epsilon = 0.2
gamma = 0.95
learn_rate = 0.5
max_episodes = 700
max_steps = 2000
trials = 50


# with open('sarsa_q.dict', 'rb') as file:
#     starting_Q = pickle.load(file)
trial_rewards = []
for i in range(trials):
    print(f"Trial: {i}")
    env = SymbolicObsWrapper(gym.make("MiniGrid-FourRooms-v0", max_steps=max_steps))
    starting_Q = {}
    trained_Q, steps_arr, reward_arr = train(env, Q=starting_Q)
    trial_rewards.append(reward_arr)

with open('eps_trials.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows(trial_rewards)

with open('sarsa_q_no_random.dict', 'wb') as file:
     pickle.dump(trained_Q, file)
print(trained_Q)

# print(steps_arr)
# print(reward_arr)

# with open('learning.csv', 'w') as file:
#     writer = csv.writer(file)
#     writer.writerow(steps_arr)
#     writer.writerow(reward_arr)


