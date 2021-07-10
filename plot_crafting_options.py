import matplotlib.pyplot as plt
import numpy as np

from crafting.examples.minecraft.world import McWorld
from options_metrics.complexity import get_used_nodes, general_complexity, get_nodes_types_lists

MC_WORLD = McWorld()
ALL_OPTIONS = MC_WORLD.get_all_options()

options_keys = np.array(list(ALL_OPTIONS.keys()))
all_complexities, all_used_nodes = get_used_nodes(ALL_OPTIONS)
nodes_by_type = get_nodes_types_lists(list(ALL_OPTIONS.values()))
# for node_list, nodes_type in zip(nodes_by_type, ("Action", "Features", "Options")):
#     print(nodes_type)
#     print(node_list)
# print(list(all_used_nodes.keys()))

options_complexities = np.array([all_complexities[option_key] for option_key in options_keys])
options_learning_complexities = np.array([
    np.array(general_complexity(option_key, nodes_by_type, all_used_nodes))
    for option_key in options_keys])
complexity_rank = np.argsort(options_complexities)

print("TotalComplexity\t| Complexity\t| SavedComplexity\t| Option")
print("------------------------------------------------------")
for rank in complexity_rank:
    option_name = options_keys[rank]
    option = ALL_OPTIONS[option_name]
    title = str(option_name)
    complexity = options_complexities[rank]
    learning_complexity = options_learning_complexities[rank][0]
    saved_complexity = options_learning_complexities[rank][1]
    print(f"{complexity}\t\t| {learning_complexity}\t\t| {saved_complexity}\t\t| {title}")

    # if hasattr(option, 'draw_graph'):
    #     title += f" - Complexity:{complexity}"
    #     fig, ax = plt.subplots()
    #     option.draw_graph(ax)
    #     plt.title(title)
    #     plt.show()
