import os
import matplotlib.pyplot as plt

import networkx as nx

from option_graph.requirements_graph import build_requirement_graph
from option_graph.layouts.metabased import leveled_layout_energy

from crafting.examples.minecraft.world import McWorld

MC_WORLD = McWorld()
ALL_OPTIONS = MC_WORLD.get_all_options()
options_list = list(ALL_OPTIONS.values())

requirement_graph = build_requirement_graph(options_list, verbose=1)

for node in requirement_graph.nodes():
    print(node, requirement_graph.nodes[node]['level'])

pos = leveled_layout_energy(requirement_graph)

nx.draw_networkx(requirement_graph, pos)
plt.show()
