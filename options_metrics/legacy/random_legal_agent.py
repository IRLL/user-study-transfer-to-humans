from time import time
import numpy as np

from crafting.examples import MineCraftingEnv
from learnrl import Agent, Playground

class RandomLegalAgent(Agent):

    def act(self, observation, greedy=False):
        _, action_is_legal = observation
        legal_actions = np.where(action_is_legal)[0]
        return np.random.choice(legal_actions)

env = MineCraftingEnv(
    max_step=50, 
    observe_legal_actions=True,
    tasks=['obtain_diamond'],
    tasks_can_end=[True],
    verbose=0,
    fps=240
)

agent = RandomLegalAgent()

playground = Playground(env, agent)
playground.run(100, verbose=1, render=False, episodes_cycle_len=100)

print(f'Player inventory at end: {env.player.inventory}')
