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
# Import other libs
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pickle

class FirstVisitMonteCarlo:
    """First Visit Monte Carlo Control for the Four Rooms environment."""
    def __init__(self, env, Q={}):
        self.env = env
        self.gamma = 0.95
        self.max_episodes = 300
        self.max_steps = 5000
        self.epsilon = 0.2
        self.Q = Q
        # Dictionary to sum returns for each state.
        self.Returns = defaultdict(list)
    
    def check_state(self, s):
        if s not in self.Q:
            self.Q[s] = np.zeros((4,1))

    def pick_action(self, state):
        """Selects an action based on the epsilon-greedy policy."""
        # Exploitation: 
        if np.random.rand() < self.epsilon:
            return random.randint(0, 3)
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
                reward = reward if done else -0.5
                episode_state.append((state, action, reward))
                state = state2
                step += 1

            G = 0
            visited = set()
            for t, (state, action, reward) in enumerate(episode_state):
                G = self.gamma * G + reward
                if (state, action) not in visited:
                    visited.add((state, action))
                    # if state not in self.Q:
                    #     self.Q[state] = np.zeros((4))
                    
                    if (state, action) not in self.Returns:
                        self.Returns[(state, action)] = []

                    self.Returns[(state, action)].append(G)

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

if __name__ == "__main__":
    rewards_arr = []

    for trial in range(50):
        # Create and initialize the environment
        env = gym.make("MiniGrid-FourRooms-v0")
        # env = gym.make("MiniGrid-FourRooms-v0", render_mode='human')
        env = SymbolicObsWrapper(env)
        
        try:
            with open('firstvisitmc.dict', 'rb') as file:
                starting_Q = pickle.load(file)
                os.remove('firstvisitmc.dict')
        except FileNotFoundError:
            starting_Q = {}

        # Initialize the class
        montecarlo = FirstVisitMonteCarlo(env, starting_Q)
        trained_Q, steps_arr, rewards_pre_episode = montecarlo.train(env)
        rewards_arr.append(rewards_pre_episode)

        with open('firstvisitmc.dict', 'wb') as file:
            pickle.dump(trained_Q, file)
        
        print(f'Trial {trial + 1} completed.')
    
    # Plot the results
    plot_graph(rewards_arr)