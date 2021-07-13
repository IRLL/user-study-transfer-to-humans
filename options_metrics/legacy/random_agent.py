from time import time
import numpy as np

from crafting.examples import MineCraftingEnv
from learnrl import Agent, Playground

class RandomAgent(Agent):

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, greedy=False):
        return self.action_space.sample()

env = MineCraftingEnv(max_step=200, tasks=['obtain_diamond'], tasks_can_end=[True])
agent = RandomAgent(env.action_space)
playground = Playground(env, agent)
playground.run(100, verbose=5, render=False)
print(f'Player inventory at end: {env.player.inventory}')