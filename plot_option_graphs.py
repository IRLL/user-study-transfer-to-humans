import os
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

from crafting.examples.minecraft.world import McWorld
from options_metrics.complexity import get_used_nodes, general_complexity, get_nodes_types_lists
from options_metrics.binary_utility import binary_graphbased_utility

MC_WORLD = McWorld()
ALL_OPTIONS = MC_WORLD.get_all_options()

for option_name, option in ALL_OPTIONS.items():
    if hasattr(option, 'draw_graph') and 'Stick' in option_name:
        options_images_path = os.path.join('images', 'options_graphs')
        if not os.path.exists(options_images_path):
            os.makedirs(options_images_path)
        fig, ax = plt.subplots()
        fig.set_facecolor('#181a1b')
        ax.set_facecolor('#181a1b')
        option.draw_graph(ax, fontcolor='white')
        ax.set_axis_off()
        dpi = 96
        width, height = (1056, 719)
        fig.set_size_inches(width/dpi, height/dpi)
        plt.tight_layout()
        plt.title(option_name)
        plt.show()
