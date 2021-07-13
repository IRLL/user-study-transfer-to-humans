from typing import List, Dict
from options_metrics.option import Option

def binary_graphbased_utility(option:Option, solving_options:List[Option],
        used_nodes:Dict[str, Dict[str, int]]) -> bool:
    """ Returns if the option in the option graph of any solving_option.
    
    Args:
        option: option of which we want to compute the utility.
        solving_options: list of options that solves the task of interest.
        used_nodes: dictionary mapping option_id to nodes used in the option.
    
    Returns:
        True if the option in the option graph of any solving_option. False otherwise.

    """

    for solving_option in solving_options:
        if option.option_id == solving_option.option_id:
            return True
        if option.option_id in used_nodes[solving_option.option_id] and\
            used_nodes[solving_option.option_id][option.option_id] > 0:
            return True
    return False
