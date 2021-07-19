import networkx as nx
import matplotlib.patches as mpatches
from matplotlib.axes import Axes
from matplotlib.legend_handler import HandlerPatch

from options_metrics.graph import option_layout, draw_networkx_nodes_images

class OptionGraph(nx.DiGraph):

    def add_node_feature_condition(self, node_name:str, image):
        self.add_node(node_name, type='feature_check', color='blue', image=image)

    def add_node_option(self, node_name:str, image):
        self.add_node(node_name, type='option', color='orange', image=image)

    def add_node_action(self, node_name:str, image):
        self.add_node(node_name, type='action', color='red', image=image)

    def add_node_empty(self, node_name:str):
        self.add_node(node_name, type='empty', color='purple', image=None)

    def add_edge_conditional(self, u_of_edge, v_of_edge, is_yes:bool):
        color = 'green' if is_yes else 'red'
        self.add_edge(u_of_edge, v_of_edge, type='conditional', color=color)

    def add_edge_any(self, u_of_edge, v_of_edge):
        self.add_edge(u_of_edge, v_of_edge, type='any', color='purple')

    def add_predecessors(self, prev_checks, node, force_any=False):
        if len(prev_checks) > 1 or (force_any and len(prev_checks) > 0):
            for pred in prev_checks:
                self.add_edge_any(pred, node)
        elif len(prev_checks) == 1:
            self.add_edge_conditional(prev_checks[0], node, True)

    def draw(self, ax:Axes) -> Axes:
        if len(list(self.nodes())) > 0:
            pos = option_layout(self)

            draw_networkx_nodes_images(self, pos, ax=ax, img_zoom=0.5)

            # nx.draw_networkx_labels(
            #     self.graph, pos,
            #     ax=ax,
            #     font_color= 'white' if background_color='black' else 'black',
            # )

            nx.draw_networkx_edges(
                self, pos,
                ax=ax,
                arrowsize=20,
                arrowstyle="-|>",
                min_source_margin=0, min_target_margin=10,
                node_shape='s', node_size=1500,
                edge_color=[color for _, _, color in self.edges(data='color')]
            )

            legend_patches = [
                mpatches.Patch(facecolor='none', edgecolor='blue', label='Feature condition'),
                mpatches.Patch(facecolor='none', edgecolor='orange', label='Option'),
                mpatches.Patch(facecolor='none', edgecolor='red', label='Action'),
            ]
            legend_arrows = [
                mpatches.FancyArrow(0, 0, 1, 0,
                    facecolor='green', edgecolor='none', label='Condition (True)'),
                mpatches.FancyArrow(0, 0, 1, 0,
                    facecolor='red', edgecolor='none', label='Condition (False)'),
                mpatches.FancyArrow(0, 0, 1, 0,
                    facecolor='purple', edgecolor='none', label='Any'),
            ]

            # Draw the legend
            ax.legend(
                handles=legend_patches + legend_arrows,
                handler_map={
                    # Patch arrows with fancy arrows in legend
                    mpatches.FancyArrow : HandlerPatch(
                        patch_func=lambda width, height, **kwargs:mpatches.FancyArrow(
                            0, 0.5*height, width, 0, width=0.2*height,
                            length_includes_head=True, head_width=height, overhang=0.5
                        )
                    ),
                }
            )

        return ax


class Option():

    """ Abstract class for options """

    def __init__(self, option_id) -> None:
        self.option_id = option_id
        self._graph = None

    def __call__(self, observations, greedy: bool=False):
        """ Use the option to get next actions.

        Args:
            observations: Observations of the environment.
            greedy: If true, the agent should act greedily.

        Returns:
            actions: Actions given by the option with current observations.
            option_done: True if the option is done, False otherwise.

        """
        raise NotImplementedError

    def interest(self, observations) -> float:
        raise NotImplementedError

    def build_graph(self) -> OptionGraph:
        raise NotImplementedError

    def draw_graph(self, ax:Axes) -> Axes:
        return self.graph.draw(ax)

    @property
    def graph(self) -> OptionGraph:
        if self._graph is None:
            self._graph = self.build_graph()
        return self._graph
