from options_metrics import complexity

from crafting.examples import MineCraftingEnv
from crafting.examples.minecraft.items import ENCHANTING_TABLE, STICK, WOODEN_PICKAXE
from crafting.examples.minecraft.rendering import get_human_action

env = MineCraftingEnv(verbose=1, max_step=100,
    tasks=['obtain_enchanting_table'], tasks_can_end=[True]
)

ALL_GET_OPTIONS = env.world.get_all_options()
enchant_table_option = ALL_GET_OPTIONS[ENCHANTING_TABLE.item_id]
saved_complexities = {}
for option_id in ALL_GET_OPTIONS:
    print(option_id, complexity(ALL_GET_OPTIONS[option_id], ALL_GET_OPTIONS)[0])

# for _ in range(2):
#     observation = env.reset()
#     done = False
#     total_reward = 0
#     while not done:
#         rgb_array = env.render(mode='rgb_array')

#         enchant_action_id, _ = enchant_table_option(observation)
#         print(f'For Enchanting Table: {env.action_from_id(enchant_action_id)}')

#         action = get_human_action(env,
#             # additional_events=additional_events,
#             **env.render_variables
#         )
#         action_id = env.action(*action)
#         print(f'Human did: {env.action_from_id(action_id)}')

#         observation, reward, done, infos = env(action_id)
#         total_reward += reward

#     print("SCORE: ", total_reward)
