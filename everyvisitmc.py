#########################################################################
# Description: First Visit Monte Carlo Implementation for the Four Rooms
# Update:
    # Date: May 10-12, 2025: First Implementation
    # Date: May 30, 2025: 
#########################################################################
# Import minigrid lib
import minigrid as mg
import gymnasium as gym
# Import defaultdict for dictionary with default values
from collections import defaultdict
# Import symbolic observation wrapper
from minigrid.wrappers import SymbolicObsWrapper
from gymnasium.wrappers import TimeLimit

from minigrid.core.constants import DIR_TO_VEC
from minigrid.core.world_object import Wall
from minigrid.core.world_object import Goal

# Import other libs
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv

def randomize(env):
    """Randomly place the agent and the goal on different free (non-wall) tiles."""
    env = env.unwrapped  # Access the raw MiniGrid environment
    width, height = env.width, env.height

    # Collect all valid (non-wall, non-object) positions
    free_positions = []
    for x in range(width):
        for y in range(height):
            cell = env.grid.get(x, y)
            if cell is None:
                free_positions.append((x, y))

    # Randomly select positions (agent and goal must be different)
    positions = random.sample(free_positions, 2)
    agent_pos, goal_pos = positions

    # Set agent
    agent_dir = random.randint(0, 2)
    env.agent_pos = agent_pos
    env.agent_dir = agent_dir

    # Clear any existing goal(s)
    for x in range(width):
        for y in range(height):
            obj = env.grid.get(x, y)
            if obj and obj.type == 'goal':
                env.grid.set(x, y, None)

    # Set new goal
    env.grid.set(*goal_pos, Goal())
    env.goal_pos = goal_pos

class FirstVisitMonteCarlo:
    """First Visit Monte Carlo Control for the Four Rooms environment."""
    def __init__(self, env, Q={}):
        self.env = env
        self.gamma = 0.95
        self.max_episodes = 700
        self.max_steps = 5000
        self.epsilon = 0.2
        self.Q = Q
        # Dictionary to sum returns for each state.
        self.Returns = defaultdict(list)
    
    def check_state(self, s):
        if s not in self.Q:
            self.Q[s] = np.zeros((3,1))

    def pick_action(self, state):
        """Selects an action based on the epsilon-greedy policy."""
        # Exploitation: 
        if np.random.rand() < self.epsilon:
            return random.randint(0, 2)
        else:
            # Select the best action based on Q values
            return np.argmax(self.Q[state][:])
    
    def train(self, env):
        episode = 0
        steps_arr = []
        rewards_per_episode = []

        while episode < self.max_episodes:
            # For training purpose, set env to the specific state:
            # obs, _ = env.reset()
            obs, _ = env.reset(seed=env.np_random_seed)
            # randomize(env)

            step = 0
            done = False
            # Generate an episode:
            episode_state = []

            # state = (obs['direction'], tuple(obs['image'].flatten()))
            state = (obs['direction'], obs['image'].data.tobytes())

            while not done and step < self.max_steps:
                self.check_state(state)
                action = self.pick_action(state)
                obs2, reward, done, _, _ = env.step(action)
                # env.render()
                # state2 = (obs2['direction'], tuple(obs2['image'].flatten()))
                state2 = (obs2['direction'], obs2['image'].data.tobytes())
                # Apeend the state, action, and reward to the episode
                # reward = reward if done else -0.5
                episode_state.append((state, action, reward))
                state = state2
                step += 1

            G = 0
            visited = set()
            for t in reversed(range(len(episode_state))):       
                state, action, reward = episode_state[t]
                G = self.gamma * G + reward

                if state not in self.Q:
                    self.Q[state] = np.zeros((3))

                if (state, action) not in self.Returns:
                    self.Returns[(state, action)] = []

                self.Returns[(state, action)].append(G)
                    # Update the Q value
                self.Q[state][action] = np.mean(self.Returns[(state, action)])

            episode_reward = sum([r for _, _, r in episode_state])
            rewards_per_episode.append(episode_reward)

            print(f'episode {episode}, Done: {done}, Steps: {step}')
            steps_arr.append(step)
            episode += 1
    
        return (self.Q, steps_arr, rewards_per_episode)

def plot_graph(all_rewards):
    """Plots the graph of steps per episode."""
    # Convert to NumPy array: shape (num_trials, num_episodes)
    all_rewards_np = np.array(all_rewards)

    # Compute mean and std across trials
    mean_rewards = np.mean(all_rewards_np, axis=0)
    std_rewards = np.std(all_rewards_np, axis=0)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(mean_rewards, label="Mean Reward", linewidth=2)
    # plt.fill_between(range(len(mean_rewards)),
    #                 mean_rewards - std_rewards,
    #                 mean_rewards + std_rewards,
    #                 alpha=0.3)
    plt.fill_between(range(len(mean_rewards)),
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Learning Curve Averaged Over 50 Trials")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("learning_curve.png")
    plt.show()

# def plot_graph(all_rewards, epsilon_values):
#     """
#     Plots reward per episode for each epsilon (single trial per epsilon).
    
#     Parameters:
#     - all_rewards: List of 1D lists (each list is reward per episode for one epsilon)
#     - epsilon_values: List of epsilon values (floats) corresponding to each reward list
#     """
#     plt.figure(figsize=(12, 6))

#     for rewards, eps in zip(all_rewards, epsilon_values):
#         if len(rewards) == 0:
#             continue
#         plt.plot(rewards, label=f"ε = {eps:.1f}", linewidth=2)

#     plt.xlabel("Episode")
#     plt.ylabel("Reward")
#     plt.title("Learning Curve per Epsilon (1 Trial Each)")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("learning_curve_single_trial.png")
#     plt.show()


if __name__ == "__main__":
    rewards_arr = []

    for trial in range(5):
        # Create and initialize the environment
        env = SymbolicObsWrapper(gym.make("MiniGrid-FourRooms-v0", max_steps=5000))

        
        try:
            with open('everyvisitmc.dict', 'rb') as file:
                starting_Q = pickle.load(file)
                # os.remove('firstvisitmc.dict')
        except FileNotFoundError:
            starting_Q = {}

        # Initialize the class
        montecarlo = FirstVisitMonteCarlo(env, starting_Q)
        trained_Q, steps_arr, rewards_pre_episode = montecarlo.train(env)
        rewards_arr.append(rewards_pre_episode)

        with open('everyvisitmc.dict', 'wb') as file:
            pickle.dump(trained_Q, file)
        
        print(f'Trial {trial + 1} completed.')
    
    # Plot the results
    plot_graph(rewards_arr)

# if __name__ == "__main__":
#     rewards_eps = []
#     epsilon = 0 # Initial epsilon value
#     for i in range(1, 11):
#         rewards_trials = []
#         epsilon = i / 10.0
#         print(f"Starting training with epsilon: {epsilon}")
#         for trial in range(3):
#             # Create and initialize the environment
            # env = SymbolicObsWrapper(gym.make("MiniGrid-FourRooms-v0", max_steps=5000))
            
#             try:
#                 with open('firstvisitmc.dict', 'rb') as file:
#                     starting_Q = pickle.load(file)
#                     os.remove('firstvisitmc.dict')
#             except FileNotFoundError:
#                 starting_Q = {}

#             # Initialize the class
#             montecarlo = FirstVisitMonteCarlo(env, starting_Q)
#             trained_Q, steps_arr, rewards_per_episode = montecarlo.train(env, epsilon)
#             # Append the rewards array to the rewards array collection
#             rewards_trials.append(rewards_per_episode)

#             with open('firstvisitmc.dict', 'wb') as file:
#                 pickle.dump(trained_Q, file)
            
#             print(f'Trial {trial + 1} completed.')
        
#         rewards_eps.append(np.average(rewards_trials, axis=0))
#         epsilon += 0.1
    
#     with open('mcfv_epsilon.csv', 'w') as file:
#         writer = csv.writer(file)
#         writer.writerows(rewards_eps)