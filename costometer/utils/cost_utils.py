"""These utilities are related to cost: parameter strings, combinations and q-values"""
import time
from itertools import product

import dill as pickle
import numpy as np
from mouselab.cost_functions import linear_depth
from mouselab.exact_utils import timed_solve_env
from mouselab.graph_utils import annotate_mdp_graph
from mouselab.mouselab import MouselabEnv
from numpy.random import default_rng


def save_q_values_for_cost(
    experiment_setting,
    cost_function=linear_depth,
    cost_function_name: str = None,
    cost_params=None,
    ground_truths=None,
    structure=None,
    path=None,
    verbose=True,
    **env_params,
):
    """
    Finds q values for an experiment/parameter combination and saves as dictionary
    :param experiment_setting: experiment setting, e.g. high_increasing
    :param cost_function: cost function to use
    :param cost_function_name: str
    :param cost_params: inputs to cost function,
            e.g. {static_cost_weight: 1, depth_cost_weight: 0}
    :param ground_truths: ground_truths for which possible states are found and calculated  # noqa
    :param structure: structure dictionaries extracted from json (what we use for experiments on MTurk to define location of nodes)
    :param path: Pathlib location of place to save output file
    :param verbose: whether to print out progress updates
    :param env_params: kwargs, any MouselabEnv settings other than cost parameters
    :return: info dictionary which includes"
                Q dictionary (q_dictionary key), timing, parameters, etc.
    """
    # prevent mutable default argument
    if cost_params is None:
        cost_params = {}
    else:
        cost_params = cost_params

    if cost_function_name is None:
        cost_function_name = cost_function.__name__

    categorical_gym_env = MouselabEnv.new_symmetric_registered(
        experiment_setting, cost=cost_function(**cost_params), **env_params
    )
    # if info on structure provided, add to mdp graph (probably needed in cost)
    if structure:
        categorical_gym_env.mdp_graph = annotate_mdp_graph(
            categorical_gym_env.mdp_graph, structure
        )

    # solve environment
    _, _, _, info = timed_solve_env(
        categorical_gym_env, verbose=verbose, save_q=True, ground_truths=ground_truths
    )

    # add experiment and parameter settings to info dict
    info["env_params"] = env_params
    info["cost_params"] = cost_params
    info["cost_function"] = cost_function_name

    # saves Q-function
    if path is not None:
        parameter_string = get_param_string(cost_params)
        path.joinpath(f"{experiment_setting}/{cost_function_name}/").mkdir(
            parents=True, exist_ok=True
        )
        filename = path.joinpath(
            f"{experiment_setting}/{cost_function_name}/"
            f"Q_{experiment_setting}_{parameter_string}_{time.strftime('%Y%m%d-%H%M')}.pickle"  # noqa: E501
        )

        with open(filename, "wb") as f:
            pickle.dump(info, f)
    return info


def get_param_string(cost_params):
    if isinstance(cost_params, dict):
        parameter_string = "_".join(
            (
                f"{param:.2f}" if not isinstance(param, str) else param
                for key, param in sorted(cost_params.items())
            )
        )
    else:
        # assume correct ordering
        parameter_string = "_".join(
            (
                f"{param:.2f}" if not isinstance(param, str) else param
                for param in cost_params
            )
        )
    return parameter_string


def load_q_file(experiment_setting, cost_function, cost_params, path):
    """
    Load Q file given experiment / cost settings
    :param experiment_setting: experiment layout
    :param cost_function: cost function
    :param cost_params: cost parameters
    :param path: path where data is
    :return: dictionary containing q values
    """
    parameter_string = get_param_string(cost_params=cost_params)

    files = list(
        path.glob(
            f"{experiment_setting}/{cost_function.__name__}/"
            f"Q_{experiment_setting}_{parameter_string}_*.pickle"
        )  # noqa: E501
    )

    if len(files) > 1:
        print(f"Number of files: {len(files)}\n Choosing latest file.")
    # always a list, so why not sort
    filename = sorted(files, reverse=True)[0]
    with open(filename, "rb") as f:
        info = pickle.load(f)

    return info["q_dictionary"]


def save_combination_file(combinations, filename, location):
    """
    This file saving any general list of combinations
    :param combinations: list of iterables, one for each combination of parameters
    :param filename: filename to use
    :param location: where to save the file
    :return: nothing
    """
    np.savetxt(
        location.joinpath(f"{filename}.txt"),
        np.asarray(combinations),
        fmt="%.02f",
        delimiter=",",
    )


def create_parameter_grid(start, stop, step=None, num_params=2):
    """
    Creates a combination of parameters according to some grid
    :param start: First value on grid
    :param stop: The grid goes up to this value, but usually does not include it
                    (see np.arange documentation)
    :param step: Space between each set of points on the grid
    :param num_params: Number of parameters considered
    :return: a list with each combination of possible parameters as an entry
    """
    reward = np.arange(start, stop, step)
    combinations = list(product(*[reward] * num_params))

    return combinations


def create_random_parameter_grid(
    start, stop, num_params, filename=None, num_combinations=100, seed=None
):
    """
    Creates random combinations of parameters, within some range
    :param start: Start of range
    :param stop: End of range
    :param num_params: Number of parameters considered
    :param filename: filename to use when saving this file, if None do not save
    :param num_combinations: number of random combinations to sample
    :param seed: random seed
    :return: a list with each combination of possible parameters as an entry
    """
    rng = default_rng(seed)
    combinations = rng.uniform(start, stop, (num_combinations, num_params))

    if filename is not None:
        save_combination_file(combinations, filename)
    return combinations
