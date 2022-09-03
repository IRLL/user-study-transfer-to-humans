import os
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

from option_graph.metrics.complexity import learning_complexity
from option_graph.metrics.complexity.histograms import nodes_histograms
from option_graph.metrics.utility import binary_graphbased_utility

from crafting.examples.minecraft.world import McWorld

MC_WORLD = McWorld()
ALL_OPTIONS = MC_WORLD.get_all_options()
options_names = list(ALL_OPTIONS.keys())

used_nodes_all = nodes_histograms(list(ALL_OPTIONS.values()))
learning_complexities = []
saved_complexities = []
for option_name, option in tqdm(ALL_OPTIONS.items(),
        desc='Computing options complexities', total=len(ALL_OPTIONS)):
    learning_c, saved_c = learning_complexity(option, used_nodes_all)
    learning_complexities.append(np.array(learning_c))
    saved_complexities.append(np.array(saved_c))

learning_complexities = np.array(learning_complexities)
saved_complexities = np.array(saved_complexities)

complexity_rank = np.argsort(learning_complexities + saved_complexities)

diplay_names = np.array([name.split('(')[0] for name in options_names])
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
    option = ALL_OPTIONS[options_names[rank]]
    is_useful = [
        str(int(binary_graphbased_utility(option, solving_option, used_nodes_all)))
        for _, solving_option in solving_options.items()
    ]
    is_useful = "".join(is_useful)
    title = str(option_name)
    learning_c= learning_complexities[rank]
    saved_complexity = saved_complexities[rank]
    total_complexity = learning_c + saved_complexity
    print(f"{total_complexity}\t\t| {learning_c}\t\t| {saved_complexity}\t\t| {title}")

    try:
        fig, ax = plt.subplots()
        fig.set_facecolor('#181a1b')
        ax.set_facecolor('#181a1b')
        option.graph.draw(ax, fontcolor='white')
        ax.set_axis_off()

        title += f" - Learning complexity:{learning_c}"
        title += f" - Complexity:{total_complexity}"

        option_name = '_'.join(option_name.lower().split(' '))
        option_title = f'option-{int(learning_c)}-{is_useful}-{option_name}.png'

        dpi = 96
        width, height = (1056, 719)
        fig.set_size_inches(width/dpi, height/dpi)
        plt.tight_layout()
        show = False
        if show:
            plt.title(title)
            plt.show()
        else:
            options_images_path = os.path.join('images', 'options_graphs')
            os.makedirs(options_images_path, exist_ok=True)
            save_path = os.path.join(options_images_path, option_title)
            plt.savefig(save_path, dpi=dpi)
            plt.close()
    except NotImplementedError:
        continue

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
