import os
import matplotlib.pyplot as plt

from tqdm import tqdm

from option_graph.metrics.complexity import learning_complexity
from option_graph.metrics.complexity.histograms import nodes_histograms
from option_graph.requirements_graph import build_requirement_graph

from crafting.examples.minecraft.world import McWorld

MC_WORLD = McWorld()
ALL_OPTIONS = MC_WORLD.get_all_options()
options_list = list(ALL_OPTIONS.values())
options_names = list(ALL_OPTIONS.keys())

used_nodes_all = nodes_histograms(options_list)
requirment_graph = build_requirement_graph(options_list)

relative_complexity_save = []
depths = []

for option in tqdm(options_list, desc='Computing options depth and complexities'):
    learning_c, saved_c = learning_complexity(option, used_nodes_all)
    relative_complexity_save.append(saved_c/(learning_c + saved_c))
    depths.append(requirment_graph.nodes[option]['level'])

plt.plot(depths, relative_complexity_save, linestyle='', marker='+')
plt.show()
