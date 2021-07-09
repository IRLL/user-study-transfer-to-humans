import networkx as nx


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

    def build_graph(self) -> nx.DiGraph:
        raise NotImplementedError
    
    def draw_graph(self, ax):
        raise NotImplementedError

    @property
    def graph(self) -> nx.DiGraph:
        if self._graph is None:
            self._graph = self.build_graph()
        return self._graph

