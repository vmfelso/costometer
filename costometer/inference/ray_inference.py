"""Optimization with ray[tune]."""
import itertools
import logging
from copy import deepcopy
from typing import Any, Callable, Dict, List, Type

import numpy as np
import pandas as pd
import ray
from mouselab.distributions import Categorical
from ray import tune

from scipy import stats # noqa
from costometer.agents.vanilla import Participant
from costometer.inference.base import BaseInference
from costometer.utils import get_param_string, load_q_file, traces_to_df, adjust_state, adjust_ground_truth

class BaseRayInference(BaseInference):
    """Base Ray optimization class"""

    def __init__(
        self,
        traces: List[Dict[str, List]],
        participant_class: Type[Participant],
        participant_kwargs: Dict[str, Any],
        cost_function: Callable,
        cost_parameters: Dict[str, Categorical],
        cost_function_name: str = None,
        held_constant_policy_kwargs: Dict[str, Any] = None,
        policy_parameters: Dict[str, Any] = None,
        local_mode: bool = True,
        optimization_settings: Dict[str, Any] = None,
    ):
        """

        :param traces:
        :param participant_class:
        :param participant_kwargs:
        :param cost_function:
        :param cost_parameters:
        :param cost_function_name:
        :param held_constant_policy_kwargs:
        :param policy_parameters:
        :param local_mode:
        """
        super().__init__(traces)

        self.participant_class = participant_class
        self.participant_kwargs = participant_kwargs
        self.cost_function = cost_function
        self.cost_function_name = cost_function_name

        if cost_parameters is None:
            self.cost_parameters = {}
        else:
            self.cost_parameters = cost_parameters

        # policy parameters vary between agents
        if held_constant_policy_kwargs is None:
            self.held_constant_policy_kwargs = {}
        else:
            self.held_constant_policy_kwargs = held_constant_policy_kwargs
        if policy_parameters is None:
            self.policy_parameters = {}
        else:
            self.policy_parameters = policy_parameters

        self.local_mode = local_mode

        if optimization_settings is None:
            self.optimization_settings = {}
        else:
            self.optimization_settings = optimization_settings

        self.optimization_results = None

    def function_to_optimize(self, config, trace, optimize=True):
        """

        :param config:
        :param trace:
        :param optimize:
        :return:
        """
        raise NotImplementedError

    def get_optimization_space(self):
        """

        :return:
        """
        raise NotImplementedError

    def run(self):
        """

        :return:
        """
        raise NotImplementedError

    def get_best_parameters(self):
        """

        :return:
        """
        return [result.get_best_config() for result in self.optimization_results]

    def get_output_df(self):
        """

        :return:
        """
        traces = deepcopy(self.traces)
        for best_params, trace in zip(self.get_best_parameters(), traces):
            trace["pi"] = self.function_to_optimize(best_params, trace, optimize=False)
        trace_df = traces_to_df(traces)
        return trace_df

    def get_optimization_results(self):
        """

        :return:
        """
        results_for_df = {}
        for optimization_result in self.optimization_results:
            results_for_df = {**results_for_df, **optimization_result.results}

        results_df = pd.DataFrame.from_dict(results_for_df).transpose()
        config_columns = results_df["config"].apply(pd.Series)
        return results_df.join(config_columns)


class CostRayInference(BaseRayInference):
    """"Optimization over grid"""

    def __init__(
        self,
        traces: List[Dict[str, List]],
        participant_class: Type[Participant],
        participant_kwargs: Dict[str, Any],
        cost_function: Callable = None,
        cost_parameters: Dict[str, Categorical] = None,
        held_constant_policy_kwargs: Dict[str, Categorical] = None,
        held_constant_cost_kwargs: Dict[str, Categorical] = None,
        policy_parameters: Dict[str, Categorical] = None,
        local_mode: bool = False,
        optimization_settings: Dict[str, Any] = None,
    ):
        """

        :param traces:
        :param participant_class:
        :param participant_kwargs:
        :param cost_function:
        :param cost_parameters:
        :param held_constant_policy_kwargs:
        :parm held_constant_cost_kwargs:
        :param policy_parameters:
        :param local_mode:
        :param optimization_settings:
        """
        super().__init__(
            traces=traces,
            participant_class=participant_class,
            participant_kwargs=participant_kwargs,
            cost_function=cost_function,
            cost_parameters=cost_parameters,
            held_constant_policy_kwargs=held_constant_policy_kwargs,
            policy_parameters=policy_parameters,
            local_mode=local_mode,
            optimization_settings=optimization_settings,
        )

        self.held_constant_cost_kwargs = held_constant_cost_kwargs

        self.prior_probability_dict = {
            **{
                cost_parameter: eval(cost_prior["prior"])
                for cost_parameter, cost_prior in self.cost_parameters.items() if cost_parameter not in held_constant_cost_kwargs
            },
            **{
                policy_parameter: eval(policy_prior["prior"])
                for policy_parameter, policy_prior in self.policy_parameters.items() if policy_parameter not in held_constant_policy_kwargs
            },
        }
        self.optimization_space = self.get_optimization_space()

    def function_to_optimize(self, config, traces, optimize=True):
        """

        :param config:
        :param traces:
        :param optimize:
        :return:
        """
        policy_kwargs = {key: config[key] for key in self.policy_parameters.keys()}
        cost_kwargs = {key: config[key] for key in self.cost_parameters.keys()}
        additional_params = {}

        for key, val in self.held_constant_cost_kwargs.items():
            cost_kwargs[key] = val
            additional_params[key] = val

        for key, val in self.held_constant_policy_kwargs.items():
            if key == "q_function_generator":
                policy_kwargs["preference"] = val(cost_kwargs, policy_kwargs["gamma"], policy_kwargs["alpha"])
            else:
                policy_kwargs[key] = val
                additional_params[key] = val

        participant = self.participant_class(
            **self.participant_kwargs,
            num_trials=max([len(trace["actions"]) for trace in traces]),
            cost_function=self.cost_function,
            cost_kwargs=cost_kwargs,
            policy_kwargs={key : val for key, val in policy_kwargs.items() if key not in ["gamma", "alpha"]},
        )

        result = []
        for trace in traces:
            trace["states"] = [[adjust_state(state, policy_kwargs["alpha"], policy_kwargs["gamma"], participant.mouselab_envs[0].mdp_graph.nodes.data("depth"), True) for state in trial] for trial in trace["states"]]
            trace["ground_truth"] = [adjust_state(ground_truth, policy_kwargs["alpha"], policy_kwargs["gamma"],
                                             participant.mouselab_envs[0].mdp_graph.nodes.data("depth"), True) for ground_truth in trace["ground_truth"]]

            # participant.agent.env
            participant_likelihood = participant.compute_likelihood(trace)

            prior_prob = np.sum(
                    [
                        prior_function(config[param])
                        for param, prior_function in self.prior_probability_dict.items()
                    ]
            )
            if optimize is True:
                # sum over actions in trial, then trials
                trial_mles = np.fromiter(map(sum, participant_likelihood), dtype=float)
                mle = np.sum(trial_mles)
                map_val = mle + prior_prob

                # save mles for blocks (if they exist)
                block_mles = {}
                if "block" in trace:
                    block_indices = {
                        block: [curr_block == block for curr_block in trace["block"]]
                        for block in np.unique(trace["block"])
                    }
                    block_mles = {
                        f"{block}_map": np.sum(trial_mles[block_indices[block]]) + prior_prob
                        for block in block_indices.keys()
                    }
                # simulated trace, save info used to simulate data
                trace_info = {key: trace[key] for key in trace.keys() if "sim_" in key}
                result.append(
                    {
                        "participant_likelihood": participant_likelihood,
                        "map_val": map_val,
                        "mle": mle,
                        "trace_pid": trace["pid"][0],
                        **block_mles,
                        **trace_info,
                        **config,
                        **additional_params
                    }
                )
            else:
                result.append(participant_likelihood)

        if optimize is False:
            return result
        else:
            return {
                "map_val": np.sum([res["test_map"] for res in result]),
                "result": result,
            }

    def get_optimization_space(self):
        """

        :return:
        """

        search_space = {
            **{
                cost_parameter: eval(cost_prior["search_space"])
                for cost_parameter, cost_prior in self.cost_parameters.items()
            },
            **{
                policy_parameter: eval(policy_prior["search_space"])
                for policy_parameter, policy_prior in self.policy_parameters.items()
            },
        }
        return search_space

    def run(self):
        """

        :return:
        """
        ray.init(logging_level=logging.ERROR, local_mode=self.local_mode)
        opt_results = tune.run(
            lambda config: self.function_to_optimize(config, traces=self.traces),
            config=self.optimization_space,
            metric="map_val",
            mode="max",
            **self.optimization_settings,
        )
        self.optimization_results = [
            item
            for opt_result in opt_results.results.values()
            for item in opt_result["result"]
        ]
        ray.shutdown()

    def get_best_parameters(self):
        """

        :return:
        """
        optimization_results = self.get_optimization_results()

        sim_cols = [col for col in list(optimization_results) if "sim_" in col]
        best_param_rows = optimization_results.iloc[
            optimization_results.groupby(["trace_pid"] + sim_cols).idxmax()["mle"]
        ]

        best_results = []
        for trace in self.traces:
            best_row = best_param_rows[
                np.all(
                    best_param_rows[sim_cols]
                    == [trace[sim_col] for sim_col in sim_cols],
                    axis=1,
                )
                & (best_param_rows["trace_pid"] == trace["pid"][0])
            ]
            assert len(best_row) == 1
            best_results.extend(best_row.to_dict("records"))
        return [
            {
                key: trace[key]
                for key in {**self.policy_parameters, **self.cost_parameters}
            }
            for trace in best_results
        ]

    def get_output_df(self):
        """

        :return:
        """
        traces = deepcopy(self.traces)
        for best_params, trace in zip(self.get_best_parameters(), traces):
            trace["pi"] = self.function_to_optimize(
                best_params, traces=[trace], optimize=False
            )[0]
        trace_df = traces_to_df(traces)
        return trace_df

    def get_optimization_results(self):
        """

        :return:
        """
        return pd.DataFrame(self.optimization_results)