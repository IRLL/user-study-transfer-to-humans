from time import time

from crafting.examples import MineCraftingEnv
from crafting.examples.minecraft.rendering import get_human_action
from learnrl import Agent, Playground

class HumanAgent(Agent):

    def __init__(self, env):
        self.env = env

    def act(self, observation, greedy=False):
        action = get_human_action(self.env, *self.env.render_variables)
        action_id = self.env.action(*action)
        return action_id

env = MineCraftingEnv(
    max_step=50,
    tasks=['obtain_diamond'],
    tasks_can_end=[True],
    fps=60
)

agent = HumanAgent(env)
playground = Playground(env, agent)
playground.test(2, verbose=2)
print(f'Player inventory at end: {env.player.inventory}')