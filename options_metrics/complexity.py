from typing import Dict, List
from copy import deepcopy, copy

import numpy as np
from options_metrics import Option

def update_sum_dict(dict1, dict2):
    dict1, dict2 = deepcopy(dict1), deepcopy(dict2)
    for key, val in dict2.items():
        if key in dict1:
            if isinstance(val, dict):
                dict1[key] = update_sum_dict(dict1[key], val)
            else:
                dict1[key] += val
        else:
            dict1[key] = val
    return dict1

def linear_g(k):
    return k

def resused_without_cost_utility(k, p):
    return max(0, min(k, p + k - 1))

def complexity(option:Option, options:List, used_nodes:Dict[str, int]=None, 
        actions_complexities=1, features_complexities=1,
        _options_in_search=None, _saved_options_complexities=None):

    try:
        graph = option.get_graph()
    except NotImplementedError:
        return 0, used_nodes

    if _options_in_search is None:
        _options_in_search = []
    _options_in_search.append(option.option_id)

    if _saved_options_complexities is None:
        _saved_options_complexities = {}

    if used_nodes is None:
        used_nodes = {}

    def _complexity(option:Option, options, used_nodes):
        if option.option_id in _saved_options_complexities:
            return _saved_options_complexities[option.option_id]
        node_complexity, node_used_nodes = \
            complexity(option, options, used_nodes,
                _options_in_search=deepcopy(_options_in_search),
                _saved_options_complexities=deepcopy(_saved_options_complexities))
        _saved_options_complexities[option.option_id] = (node_complexity, node_used_nodes)
        return node_complexity, node_used_nodes
    
    def _get_node_complexity(graph, node, used_nodes):
        node_type = graph.nodes[node]['type']

        if node_type == 'action':
            node_used_nodes = {node:1}
            if isinstance(actions_complexities, dict):
                node_complexity = actions_complexities[node]
            else:
                node_complexity = actions_complexities

        elif node_type == 'feature_check':
            node_used_nodes = {node:1}
            if isinstance(features_complexities, dict):
                node_complexity = features_complexities[node]
            else:
                node_complexity = features_complexities

        elif node_type == 'option':
            _option = options[graph.nodes[node]['option_key']]

            if _option.option_id in _options_in_search:
                node_used_nodes = {}
                node_complexity = np.inf
            else:
                node_complexity, node_used_nodes = \
                    _complexity(_option, options, used_nodes)

        elif node_type == 'empty':
            node_used_nodes = {}
            node_complexity = 0

        else:
            raise ValueError(f"Unkowned node type {node_type}")

        return node_complexity, node_used_nodes

    nodes_by_level = graph.graph['nodes_by_level']
    depth = graph.graph['depth']

    complexities = {}
    nodes_used_nodes = {}

    for level in range(depth+1)[::-1]:

        for node in nodes_by_level[level]:

            node_complexity = 0
            node_used_nodes = {}

            or_complexities = []
            or_succs = []
            for succ in graph.successors(node):
                succ_complexity = complexities[succ]
                if graph.edges[node, succ]['type'] == 'any':
                    or_complexities.append(succ_complexity)
                    or_succs.append(succ)
                else:
                    node_complexity += succ_complexity
                    node_used_nodes = update_sum_dict(node_used_nodes, nodes_used_nodes[succ])

            if len(or_succs) > 0:
                min_succ_id = np.argmin(or_complexities)
                min_succ = or_succs[min_succ_id]
                min_complex = or_complexities[min_succ_id]

                node_complexity += min_complex
                node_used_nodes = update_sum_dict(node_used_nodes, nodes_used_nodes[min_succ]) 

            node_used_nodes = update_sum_dict(node_used_nodes, used_nodes)
            node_only_complexity, node_only_used_options = \
                _get_node_complexity(graph, node, node_used_nodes)

            node_complexity += node_only_complexity
            node_used_nodes = update_sum_dict(node_used_nodes, node_only_used_options)

            complexities[node] = node_complexity
            nodes_used_nodes[node] = node_used_nodes

    root = nodes_by_level[0][0]
    return complexities[root], nodes_used_nodes[root]
