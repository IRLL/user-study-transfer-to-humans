from typing import Dict, List, Union
from copy import deepcopy, copy

import numpy as np
from options_metrics.option import Option

def update_sum_dict(dict1, dict2):
    dict1, dict2 = deepcopy(dict1), deepcopy(dict2)
    for key, val in dict2.items():
        try:
            dict1[key] += val
        except KeyError:
            dict1[key] = val
    return dict1

def resused_without_cost_utility(node, k, p):
    return max(0, min(k, p + k - 1))

def linear_k_complexity(node, k):
    return k

def general_complexity(option_id, nodes_by_type, used_nodes_all:Dict[str, Dict[str, int]],
        previous_used_nodes=None,
        utility=resused_without_cost_utility,
        kcomplexity=linear_k_complexity,
        individual_complexities:Union[dict, float]=1.):

    action_nodes, feature_nodes, options_nodes = nodes_by_type
    previous_used_nodes = previous_used_nodes if previous_used_nodes else {}
    if not isinstance(individual_complexities, dict):
        individual_complexities = init_individual_complexities(
            action_nodes, feature_nodes, individual_complexities)

    total_complexity = 0
    saved_complexity = 0

    if option_id in used_nodes_all[option_id]:
        used_nodes_all[option_id].pop(option_id)

    for node in used_nodes_all[option_id]:
        n_used = used_nodes_all[option_id][node]
        n_previous_used = previous_used_nodes[node] if node in previous_used_nodes else 0

        if node in options_nodes:
            util = utility(node, n_used, n_previous_used)
            if util > 0:
                option_complexity, _ = general_complexity(
                    node, nodes_by_type,used_nodes_all,
                    previous_used_nodes=deepcopy(previous_used_nodes),
                    utility=utility, kcomplexity=kcomplexity
                )
                saved_complexity += option_complexity * util
        else:
            total_complexity += individual_complexities[node] * kcomplexity(node, n_used)

        previous_used_nodes = update_sum_dict(previous_used_nodes, {node: n_used})

    return total_complexity - saved_complexity, saved_complexity

def get_used_nodes(options:Dict[str, Option], used_nodes:Dict[str, int]=None, 
        individual_complexities:Union[dict, float]=1.):
    complexities, used_nodes = {}, {}
    for option_key, option in options.items():
        if option_key not in complexities:
            complexity, _used_nodes = get_used_nodes_single_option(option, options, 
                individual_complexities=individual_complexities)
            complexities[option_key] = complexity
            used_nodes[option_key] = _used_nodes
    return complexities, used_nodes

def init_individual_complexities(action_nodes, feature_nodes,
    individual_complexities:Union[dict, float]=1.):
    
    if isinstance(individual_complexities, (float, int)):
        individual_complexities = {
            node:individual_complexities for node in action_nodes + feature_nodes
        }
    
    elif isinstance(individual_complexities, dict):
        assert all(node in individual_complexities for node in action_nodes + feature_nodes), \
            "Individual complexities must be given for fundamental actions and features conditions"
    
    return individual_complexities


def get_nodes_types_lists(options:List[Option]):
    action_nodes, feature_nodes, option_nodes = [], [], []
    for option in options:
        try:
            graph = option.graph
        except NotImplementedError:
            continue

        for node in graph.nodes():
            node_type = graph.nodes[node]['type']
            if node_type == 'action' and node not in action_nodes:
                action_nodes.append(node)
            elif node_type == 'option' and node not in option_nodes:
                option_nodes.append(node)
            elif node_type == 'feature_check' and node not in feature_nodes:
                feature_nodes.append(node)
    return action_nodes, feature_nodes, option_nodes

def get_used_nodes_single_option(option:Option, options:Dict[str, Option],
        used_nodes:Dict[str, int]=None, individual_complexities:Union[dict, float]=1.,
        return_all_nodes=False, _options_in_search=None):

    try:
        graph = option.graph
    except NotImplementedError:
        return 0, used_nodes

    if _options_in_search is None:
        _options_in_search = []
    _options_in_search.append(option.option_id)

    if used_nodes is None:
        used_nodes = {}
    
    action_nodes, feature_nodes, _ = get_nodes_types_lists(list(options.values()))
    individual_complexities = init_individual_complexities(
        action_nodes, feature_nodes, individual_complexities)

    def _get_node_complexity(graph, node, used_nodes):
        node_type = graph.nodes[node]['type']

        if node_type in ('action', 'feature_check'):
            node_used_nodes = {node:1}
            node_complexity = individual_complexities[node]

        elif node_type == 'option':
            _option = options[node]
            if _option.option_id in _options_in_search:
                node_used_nodes = {}
                node_complexity = np.inf
            else:
                node_complexity, node_used_nodes = \
                    get_used_nodes_single_option(_option, options, used_nodes,
                        _options_in_search=deepcopy(_options_in_search))

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
    nodes_used_nodes[root] = update_sum_dict(nodes_used_nodes[root], {option.option_id: 1})
    if return_all_nodes:
        return complexities, nodes_used_nodes
    return complexities[root], nodes_used_nodes[root]
