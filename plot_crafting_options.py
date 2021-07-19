import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from crafting.examples.minecraft.world import McWorld
from options_metrics.complexity import get_used_nodes, general_complexity, get_nodes_types_lists
from options_metrics.binary_utility import binary_graphbased_utility

MC_WORLD = McWorld()
ALL_OPTIONS = MC_WORLD.get_all_options()

all_complexities, all_used_nodes = get_used_nodes(ALL_OPTIONS, verbose=1)
nodes_by_type = get_nodes_types_lists(list(ALL_OPTIONS.values()))

options_keys = np.array(list(ALL_OPTIONS.keys()))
options_complexities = np.array([all_complexities[option_key] for option_key in options_keys])

options_learning_complexities = []
desc = 'Computing options complexities'
for option_key in tqdm(options_keys, desc=desc, total=len(options_keys)):
    complexity = np.array(general_complexity(option_key, nodes_by_type, all_used_nodes))
    options_learning_complexities.append(complexity)
options_learning_complexities = np.array(options_learning_complexities)
complexity_rank = np.argsort(options_complexities)


diplay_names = np.array([name.split('(')[0] for name in options_keys])
print("TotalComplexity\t| Complexity\t| SavedComplexity\t| Option")
print("------------------------------------------------------")

solving_options = {
    'gather_wood': [ALL_OPTIONS["Get Wood(17)"]],
    'gather_stone': [ALL_OPTIONS["Get Cobblestone(4)"]],
    'obtain_book': [ALL_OPTIONS["Get Book(340)"]],
    'obtain_diamond': [ALL_OPTIONS["Get Diamond(264)"]],
    'obtain_clock': [ALL_OPTIONS["Get Clock(347)"]],
    'obtain_enchanting_table': [ALL_OPTIONS["Get Enchanting_table(116)"]],
}

for rank in complexity_rank:
    option_name = diplay_names[rank]
    option = ALL_OPTIONS[options_keys[rank]]
    is_useful = [
        str(int(binary_graphbased_utility(option, solving_options[task], all_used_nodes)))
        for task in solving_options
    ]
    utilities = "".join(is_useful)
    title = str(option_name)
    complexity = options_complexities[rank]
    learning_complexity = options_learning_complexities[rank, 0]
    saved_complexity = options_learning_complexities[rank, 1]
    print(f"{complexity}\t\t| {learning_complexity}\t\t| {saved_complexity}\t\t| {title}")

    if hasattr(option, 'draw_graph'):
        options_images_path = os.path.join('images', 'options_graphs')
        if not os.path.exists(options_images_path):
            os.makedirs(options_images_path)
        title += f" - Learning complexity:{learning_complexity}"
        title += f" - Complexity:{complexity}"
        fig, ax = plt.subplots()
        option.draw_graph(ax)
        ax.set_axis_off()

        option_name = '_'.join(option_name.lower().split(' '))
        option_title = f'option-{int(learning_complexity)}-{utilities}-{option_name}.png'
        dpi = 96
        width, height = (1056, 719)
        fig.set_size_inches(width/dpi, height/dpi)
        plt.tight_layout()
        show = False
        if show:
            plt.title(title)
            plt.show()
        else:
            save_path = os.path.join(options_images_path, option_title)
            plt.savefig(save_path, dpi=dpi)
            plt.close()

# diplay_names = diplay_names[complexity_rank]
# options_complexities = options_complexities[complexity_rank]
# options_learning_complexities = options_learning_complexities[complexity_rank]
# plt.title('Total complexity vs Learning complexity')
# plt.bar(diplay_names, options_complexities, label='total complexity')
# plt.bar(diplay_names, options_learning_complexities[:, 0], label='learning complexity')
# plt.xticks(rotation=45, ha='right')
# plt.legend()
# plt.tight_layout()
# plt.show()
