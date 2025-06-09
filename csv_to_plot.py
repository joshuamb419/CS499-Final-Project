import matplotlib.pyplot as plt
import numpy as np

# all_rewards = [[]]
with open("random_goal_and_start_100000_step.csv", 'r') as file:
    text = file.readlines()
    all_rewards = [[float(x) for x in line.split(",")] for line in text]

rewards_matrix = np.array(all_rewards)
mean_rewards = rewards_matrix.mean(axis=0)
std_rewards = rewards_matrix.std(axis=0)
episodes = range(1, len(mean_rewards) + 1)

plt.figure(figsize=(10, 6))
plt.plot(episodes, mean_rewards, label="Mean Reward", linewidth=2)
plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Learning Curve Averaged Over 50 Trials")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()