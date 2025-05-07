import minigrid as mg
import gymnasium as gym
from minigrid.manual_control import ManualControl

env = gym.make("MiniGrid-FourRooms-v0", render_mode="human")
env.reset()
# manual_control = ManualControl(env)
# manual_control.start()

actions = env.action_space

while True:
    env.step(actions.sample())
    env.render()

