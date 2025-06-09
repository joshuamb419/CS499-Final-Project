import pickle
import sys

import minigrid as mg
import gymnasium as gym
import numpy as np
import random
import csv

from gymnasium.wrappers import TimeLimit
from minigrid.wrappers import SymbolicObsWrapper
from minigrid.core.world_object import Wall
from minigrid.core.world_object import Goal

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

def unwrapState(env):
    unwrap = env.unwrapped
    return (unwrap.agent_pos, unwrap.agent_dir)

def train(env, Q = {}):
    episode = 0
    steps_arr = []
    reward_arr = []
    while episode < max_episodes:
        observation = env.reset(seed = env.np_random_seed)
        # randomize_goal_position(env)
        # randomize_agent_start(env)

        step = 0
        total_reward = 0
        done = False
        # print(observation)

        s = unwrapState(env)

        check_state(s, Q)
        a = choose_action(s, Q)

        while step < max_steps:
            # print(env.step(a))
            o2, r, done, info, info2 = env.step(a)
            s2 = unwrapState(env)
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

def randomize_goal_position(env):
    """Randomly place the goal object on a free (non-wall) tile."""
    env = env.unwrapped  # Access raw MiniGrid environment
    width, height = env.width, env.height

    # Clear existing goal
    for x in range(width):
        for y in range(height):
            obj = env.grid.get(x, y)
            if obj and obj.type == 'goal':
                env.grid.set(x, y, None)

    # Find valid empty locations
    free_positions = []
    for x in range(width):
        for y in range(height):
            obj = env.grid.get(x, y)
            if obj is None:
                free_positions.append((x, y))

    # Choose a random free location
    goal_pos = random.choice(free_positions)
    env.grid.set(*goal_pos, Goal())
    env.goal_pos = goal_pos

def randomize_agent_start(env):
    """Randomly place the agent somewhere on a free tile."""
    env = env.unwrapped
    width, height = env.width, env.height

    # Find free (non-wall, non-object) positions
    free_positions = []
    for x in range(width):
        for y in range(height):
            cell = env.grid.get(x, y)
            if cell is None or cell.type not in ['wall', 'goal']:
                free_positions.append((x, y))

    # Pick a random free position
    agent_pos = random.choice(free_positions)
    agent_dir = random.randint(0, 2)

    env.agent_pos = agent_pos
    env.agent_dir = agent_dir

epsilon = 0.6
gamma = 0.95
learn_rate = 0.5
max_episodes = 80
max_steps = 2000
trials = 10


# with open('sarsa_q.dict', 'rb') as file:
#     starting_Q = pickle.load(file)
trial_rewards = []
for i in range(trials):
    env = SymbolicObsWrapper(gym.make("MiniGrid-FourRooms-v0", max_steps=max_steps))
    starting_Q = {}
    trained_Q, steps_arr, reward_arr = train(env, Q=starting_Q)
    trial_rewards.append(reward_arr)

with open('random_goal_and_start_100000_step.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows(trial_rewards)

