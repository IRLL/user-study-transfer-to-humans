import matplotlib.pyplot as plt
import numpy as np

from crafting.examples.minecraft.world import McWorld
from options_metrics import complexity

MC_WORLD = McWorld()
ALL_OPTIONS = MC_WORLD.get_all_options()

options_keys = np.array([option_key for option_key, _ in ALL_OPTIONS.items()])
print(list(zip(ALL_OPTIONS.keys(), options_keys)))
options_complexities = np.array([
    complexity(option, ALL_OPTIONS)[0] for _, option in ALL_OPTIONS.items()])
complexity_rank = np.argsort(options_complexities)

print("Complexity \t| Option")
print("--------------------------------")
for rank in complexity_rank:
    option_key = options_keys[rank]
    try:
        option_key = int(option_key)
        option_name = MC_WORLD.item_from_id[option_key]
    except ValueError:
        option_name = option_key

    option = ALL_OPTIONS[option_key]
    title = str(option_name)
    complexity = options_complexities[rank]
    print(f"{complexity}\t\t| {title}")

    if hasattr(option, 'draw_graph'):
        title += f" - Complexity:{complexity}"
        fig, ax = plt.subplots()
        option.draw_graph(ax)
        plt.title(title)
        plt.show()
