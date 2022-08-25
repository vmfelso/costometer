from copy import deepcopy
from typing import Any, Callable, Dict, List, Type, Union

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from mcl_toolbox.mcrl_modelling.optimizer import hyperopt_space, parse_config

from costometer.agents.vanilla import Participant
from costometer.inference.base import BaseInference
from costometer.utils import traces_to_df


class MCLInference(BaseInference):
    def __init__(
        self,
        traces: List[Dict[str, List]],
        participant_class: Type[Participant],
        participant_kwargs: Dict[str, Any],
        cost_function: Callable,
        cost_parameters: Dict[str, Any],
        held_constant_policy_kwargs: Dict[str, Any] = None,
        optimization_settings: Dict[str, Any] = None,
        rng: Union[Callable, int] = None,
    ):
        """
        MCL Inference Object

        :param traces: the traces for which we are inferring parameters, \
        each a dictionary with at least "actions" and "states" as fields
        :param participant_kwargs: #TODO
        :param cost_args: #TODO
        :param hyper_opt_args: #TODO
        :param rng: #TODO
        """
        # save inputs
        self.traces = traces
        self.participant_class = participant_class
        self.participant_kwargs = deepcopy(participant_kwargs)

        # if it is, refactor it out to held_constant_policy_kwargs
        assert "params" not in self.participant_kwargs

        self.cost_function = cost_function
        self.cost_parameters = cost_parameters

        if held_constant_policy_kwargs is None:
            self.held_constant_policy_kwargs = {}
        else:
            self.held_constant_policy_kwargs = held_constant_policy_kwargs

        self.held_constant_cost = None

        if optimization_settings is None:
            self.optimization_settings = {}
        else:
            self.optimization_settings = optimization_settings

        if callable(rng):
            self.rng = rng
        else:
            self.rng = np.random.default_rng(rng)

        # best parameters
        self.best = None

        self.optimization_space = self.get_optimization_space()

    def get_optimization_space(self):
        prior_participant = self.participant_class(
            **deepcopy(self.participant_kwargs), num_trials=30
        )

        params_list = parse_config(
            prior_participant.learner,
            prior_participant.learner_attributes,
            general_params=True,
        )
        space = hyperopt_space(params_list)

        # delete extra priors for feature not used
        if len([key for key in space.keys() if "prior_" in key]) > len(
            self.participant_kwargs["features"]
        ):
            for i in range(
                len(self.participant_kwargs["features"]),
                len([key for key in space.keys() if "prior_" in key]),
            ):
                del space[f"prior_{i}"]

        # remove anything already held constant in participant_kwargs
        if len(self.held_constant_policy_kwargs) > 0:
            space = {
                key: space[key]
                for key in space.keys()
                if key not in self.held_constant_policy_kwargs.keys()
            }

        for cost_parameter, cost_prior in self.cost_parameters.items():
            vals = list(cost_prior.vals)
            if min(vals) != max(vals):
                space[cost_parameter] = hp.uniform(cost_parameter, min(vals), max(vals))
            else:
                space[cost_parameter] = min(vals)
                if self.held_constant_cost is None:
                    self.held_constant_cost = {}
                self.held_constant_cost[cost_parameter] = min(vals)

        return space

    def function_to_optimize(self, config, trace, optimize=True):
        policy_kwargs = {
            key: config[key]
            for key in config.keys()
            if key not in self.cost_parameters.keys() and "prior_" not in key
        }
        cost_kwargs = {key: config[key] for key in self.cost_parameters.keys()}

        num_features = len([key for key in config.keys() if "prior_" in key])
        policy_kwargs["priors"] = np.asarray(
            [
                config["prior_{}".format(feature_num)]
                for feature_num in range(num_features)
            ]
        )
        policy_kwargs = {**policy_kwargs, **self.held_constant_policy_kwargs}

        participant = self.participant_class(
            **deepcopy(self.participant_kwargs),
            num_trials=len(trace["actions"]),
            cost_function=self.cost_function,
            cost_kwargs=cost_kwargs,
            params=policy_kwargs,
        )

        participant_likelihood = participant.compute_likelihood(trace)

        if optimize is True:
            # sum over actions in trial, then trials
            trial_mles = np.fromiter(map(sum, participant_likelihood), dtype=float)
            mle = np.sum(trial_mles)

            # save mles for blocks
            block_mles = {}
            if "block" in trace:
                block_indices = {
                    block: [curr_block == block for curr_block in trace["block"]]
                    for block in np.unique(trace["block"])
                }
                block_mles = {
                    f"{block}_mle": np.sum(trial_mles[block_indices[block]])
                    for block in block_indices.keys()
                }

                # simulated trace, save info used to simulate data
            trace_info = {key: trace[key][0] for key in trace.keys() if "sim_" in key}

            return {
                "loss": -mle,
                "status": STATUS_OK,
                "trace_pid": trace["pid"][0],
                **trace_info,
                **block_mles,
            }
        else:
            return participant_likelihood

    def optimize(self, trace: Dict[str, List]) -> None:

        trials = Trials()
        fmin(
            lambda config: self.function_to_optimize(config, trace=trace),
            self.optimization_space,
            algo=tpe.suggest,
            show_progressbar=True,
            rstate=self.rng,
            trials=trials,
            **self.optimization_settings,
        )
        return trials

    def run(self) -> None:
        """
        Run optimization to find best parameters

        :return: None
        """

        self.trials = [self.optimize(trace) for trace in self.traces]
        self.best = [trial.argmin for trial in self.trials]

    def get_best_parameters(self) -> List[Any]:
        """
        Returns best parameters for the traces

        :return: None
        """
        return self.best

    def get_output_df(self):
        traces = deepcopy(self.traces)
        for best_params, trace in zip(self.get_best_parameters(), traces):
            trace["pi"] = self.function_to_optimize(best_params, trace, optimize=False)
        trace_df = traces_to_df(traces)
        return trace_df

    def get_optimization_results(self):
        results_for_df = []
        for trial in self.trials:
            for iter in range(len(trial.results)):
                results_for_df.append(
                    {**trial.results[iter], **trial.trials[iter]["misc"]}
                )

        results_df = pd.DataFrame.from_dict(results_for_df)
        config_columns = results_df["vals"].apply(pd.Series)
        for config_col in list(config_columns):
            config_columns[config_col] = config_columns[config_col].apply(
                lambda entry: entry[0] if len(entry) == 1 else entry
            )

        for (
            held_constant_param,
            held_constant_val,
        ) in self.held_constant_policy_kwargs.items():
            config_columns[held_constant_param] = held_constant_val

        if self.held_constant_cost is not None:
            for (
                held_constant_param,
                held_constant_val,
            ) in self.held_constant_cost.items():
                config_columns[held_constant_param] = held_constant_val

        return results_df.join(config_columns)
