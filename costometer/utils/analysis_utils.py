"""Utility functions for MAP calculation, priors and finding the best parameters."""
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Union

from statsmodels.tools.eval_measures import bic

import dill as pickle
import numpy as np
import pandas as pd
import yaml
from more_itertools import powerset
from mouselab.cost_functions import *  # noqa
from mouselab.distributions import Categorical
from mouselab.policies import RandomPolicy, SoftmaxPolicy
from scipy import stats  # noqa
from scipy.stats import rv_continuous
from statsmodels.tools.eval_measures import bic

from costometer.agents import SymmetricMouselabParticipant
from costometer.utils.cost_utils import get_param_string, load_q_file
from costometer.utils.plotting_utils import generate_model_palette
from costometer.utils.trace_utils import (
    get_trajectories_from_participant_data,
    traces_to_df,
)


def get_best_parameters(
    df: pd.DataFrame,
    cost_details: Dict[str, Any],
    priors: Dict[Any, Any],
):
    """
    Get best parameters for a dataframe, for certain cost_details

    :param df:
    :param cost_details:
    :param priors:
    :return:
    """
    best_parameter_values = {}

    # reset index's df for the indexing by best row
    df = df.reset_index()

    for prior_type, prior_dict in priors.items():
        # save best parameters for each prior
        best_parameter_values[prior_type] = {}
        # uniform should always be provided
        for subset in powerset(priors["uniform"]):
            # subset dataframe
            curr_data = df[
                df.apply(
                    lambda row: sum(
                        row[cost_param] == cost_details["constant_values"][cost_param]
                        for cost_param in list(subset)
                    )
                    == len(list(subset)),
                    axis=1,
                )
            ].copy(deep=True).reset_index(drop=True)

            # add prior
            curr_data[f"map_{prior_type}"] = curr_data.apply(
                lambda row: row["mle"]
                + sum(
                    [
                        np.log(prior_dict[param][row[param]])
                        for param in prior_dict.keys()
                        if param not in subset
                    ]
                ),
                axis=1,
            )

            # when multiple pids included,
            # some might be duplicated (e.g. pid 0 with sim cost 1 vs 2)
            sim_cols = [col for col in list(curr_data) if "sim_" in col]

            best_param_rows = curr_data.loc[  #
                curr_data.groupby(["trace_pid"] + sim_cols).idxmax(numeric_only=True)[
                    f"map_{prior_type}"
                ]
            ]
            assert np.all(
                [
                    counter == 1
                    for pid, counter in Counter(
                        best_param_rows[["trace_pid"] + sim_cols]
                        .to_records(index=False)
                        .tolist()
                    ).most_common()
                ]
            )

            best_parameter_values[prior_type][subset] = best_param_rows

    return best_parameter_values


def add_cost_priors_to_temp_priors(
    softmax_df: pd.DataFrame,
    cost_details: Dict[str, Any],
    temp_prior_details: Dict[str, Any],
    additional_params=List[str],
) -> Dict[str, Dict[str, Any]]:
    """

    :param softmax_df:
    :param cost_details:
    :param temp_prior_details:
    :param additional_params:
    :return:
    """
    full_priors = {}
    for prior, prior_inputs in temp_prior_details.items():
        priors = {}

        temp_prior = get_temp_prior(
            rv=eval(prior_inputs["rv"]),
            possible_vals=prior_inputs["possible_temps"],
            inverse=prior_inputs["inverse"],
        )
        priors["temp"] = dict(zip(temp_prior.vals, temp_prior.probs))

        non_temp_params = set(list(cost_details["constant_values"]) + additional_params) - set(["temp"])
        for cost_parameter_arg in non_temp_params:
            numeric_values = softmax_df[cost_parameter_arg][
                softmax_df[cost_parameter_arg].apply(
                    lambda entry: not isinstance(entry, str)
                )
            ]
            unique_args = np.unique(numeric_values)
            priors[cost_parameter_arg] = dict(
                zip(unique_args, np.ones(len(unique_args)) * 1 / len(unique_args))
            )

        assert np.all(
            [np.sum(priors[param_key].values()) for param_key in priors.keys()]
        )
        full_priors[prior] = priors
    return full_priors


def extract_mles_and_maps(
    data: pd.DataFrame,
    cost_details: Dict[str, Any],
    priors: Dict[Any, Any],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """

    :param data:
    :param cost_details:
    :params priors:
    :return:
    """
    # save random data first
    best_parameter_values = {}

    random_data = data[data["applied_policy"] == "RandomPolicy"].reset_index(drop=True)
    for parameter_arg in priors["uniform"].keys():
        random_data[parameter_arg] = np.nan
    best_parameter_values["RandomPolicy"] = random_data

    # now only consider softmax policy
    softmax_data = data[data["applied_policy"] == "SoftmaxPolicy"].reset_index(
        drop=True
    )

    best_parameter_values = {
        **best_parameter_values,
        "SoftmaxPolicy": get_best_parameters(softmax_data, cost_details, priors),
    }
    return best_parameter_values


def get_temp_prior(
    rv: rv_continuous, possible_vals: List[float], inverse: bool = True
) -> Categorical:
    """

    :param rv:
    :param possible_vals:
    :param inverse:
    :return:
    """
    if inverse:
        rv_vals = [1 / val for val in possible_vals]
    else:
        rv_vals = possible_vals

    normalizing_factor = sum([rv.pdf(val) for val in rv_vals])
    categorical_dist = Categorical(
        possible_vals, [rv.pdf(val) / normalizing_factor for val in rv_vals]
    )
    return categorical_dist


class AnalysisObject:
    def __init__(
        self,
        experiment_name: str,
        irl_path: Union[str, Path],
        experiment_subdirectory: Union[str, Path],
    ):
        """

        :param experiment_name:
        :param irl_path:
        """
        self.experiment_name = experiment_name

        #  must match my folder structure
        #  subfolders data/processed/{experiment_name}
        #  & analysis/{experiment_subdirectory} should exist
        self.irl_path = irl_path
        self.experiment_subdirectory = experiment_subdirectory

        # add yaml attributes to object, should be:
        # sessions, cost_functions,
        self.read_experiment_yaml()
        self.load_cost_function_details()

        if not hasattr(self, "palette_name"):
            self.palette_name = experiment_name

        if not self.simulated:
            dfs = {}
            for session in self.sessions:
                matching_files = self.irl_path.glob(
                    f"data/processed/{session}/*.csv"
                )
                for matching_file in matching_files:
                    if matching_file.stem not in dfs:
                        dfs[matching_file.stem] = [pd.read_csv(matching_file)]
                    else:
                        dfs[matching_file.stem].append(pd.read_csv(matching_file))

            self.dfs = {file_type : pd.concat(df_list) for file_type, df_list in dfs.items()}
            self.load_session_details()
        else:
            self.session_details = {
                session: {
                    "experiment_setting": session.split("/")[1],
                }
                for session in self.sessions
            }

        # only programmed correctly if all sessions have same experiment setting
        assert (
            len(
                np.unique(
                    [
                        session_details["experiment_setting"]
                        for session_details in self.session_details.values()
                    ]
                )
            )
            == 1
        )
        self.experiment_setting = [
            session_details["experiment_setting"]
            for session_details in self.session_details.values()
        ][0]

        self.optimization_data = self.load_optimization_data()

        if not self.irl_path.joinpath(
            f"analysis/{self.experiment_subdirectory}/data/"
            f"{self.experiment_name}_models_palette.pickle"
        ).is_file():
            static_palette = generate_model_palette(
                self.optimization_data["Model Name"].unique()
            )
            self.irl_path.joinpath(
                f"analysis/{self.experiment_subdirectory}/data/"
            ).mkdir(parents=True, exist_ok=True)
            with open(
                self.irl_path.joinpath(
                    f"analysis/{self.experiment_subdirectory}/data/"
                    f"{self.palette_name}_models_palette.pickle"
                ),
                "wb",
            ) as f:
                pickle.dump(static_palette, f)

    def load_cost_function_details(self):
        yaml_file = self.irl_path.joinpath(
            f"data/inputs/yamls/cost_functions/{self.cost_function}.yaml"
        )
        with open(str(yaml_file), "r") as stream:
            self.cost_details = yaml.safe_load(stream)

    def load_session_details(self):
        self.session_details = {}
        for session in self.sessions:
            yaml_file = self.irl_path.joinpath(
                f"data/inputs/yamls/experiments/{session}.yaml"
            )
            with open(str(yaml_file), "r") as stream:
                self.session_details[session] = yaml.safe_load(stream)

    def load_optimization_data(self):
        full_dfs = []
        for session in self.sessions:
            mle_and_map_files = list(
                self.irl_path.glob(
                    f"data/processed/{session}/{self.cost_function}"
                    f"/mle_and_map*.pickle"
                )
            )
            for mle_and_map_file in mle_and_map_files:
                with open(
                    mle_and_map_file,
                    "rb",
                ) as f:
                    data = pickle.load(f)
                full_dfs.extend([{**random_record, **{f"map_{prior}" : random_record["mle"] for prior in data["SoftmaxPolicy"].keys()}, "prior":"None", "model":"None", "Model Name": "Null", "session":session, "Number Parameters":0} for random_record in data["RandomPolicy"].to_dict("records")])
                for prior, prior_dict in data["SoftmaxPolicy"].items():
                    all_params = max(prior_dict, key=len)
                    for model, model_df in prior_dict.items():
                        number_parameters = len(set(all_params)-set(model))
                        cost_params_in_model = set(model).intersection(set(self.cost_details["cost_parameter_args"]))
                        additional_params_in_model = set(all_params)-set(model)-set(self.cost_details["cost_parameter_args"])

                        matching_cost_name = [cost_name for cost_params, cost_name in eval(self.cost_details["model_name"]).items() if set(cost_params) == cost_params_in_model]
                        assert(len(matching_cost_name)==1)
                        model_name = matching_cost_name[0]
                        if len(additional_params_in_model) > 0:
                            model_name = model_name + " with " + ", ".join(additional_params_in_model)
                        full_dfs.extend([{**softmax_record, "prior": prior, "model":model, "Model Name": model_name, "session":session, "Number Parameters":number_parameters} for softmax_record in model_df.to_dict("records")])
        full_df = pd.DataFrame(full_dfs)
        # delete old index column
        del full_df["index"]

        mouselab_data = self.dfs["mouselab-mdp"].copy(deep=True)
        mouselab_data["num_clicks"] = mouselab_data["num_clicks"] + 1 # add terminal action
        full_df = self.join_optimization_df_and_processed(optimization_df = full_df,
                                                          processed_df = mouselab_data[mouselab_data[ "block"] == self.block].groupby(["pid"], as_index=False).sum(),
                                                          variables_of_interest=["num_clicks"])

        full_df["bic"] = full_df.apply(
            lambda row: bic(
                llf=row[f"map_{self.prior}"],
                nobs=row["num_clicks"],
                df_modelwc=row["Number Parameters"],
            ),
            axis=1)

        return full_df

    def read_experiment_yaml(self):
        """

        :return:
        """
        yaml_file = self.irl_path.joinpath(
            f"analysis/{self.experiment_subdirectory}/"
            f"inputs/yamls/{self.experiment_name}.yaml"
        )
        with open(str(yaml_file), "r") as stream:
            yaml_dict = yaml.safe_load(stream)
        # append all entries in yaml_dict as attributes
        for key in yaml_dict:
            setattr(self, key, yaml_dict[key])

    @staticmethod
    def join_optimization_df_and_processed(
        optimization_df: pd.DataFrame, processed_df: pd.DataFrame, variables_of_interest: List[str] = None
    ) -> pd.DataFrame:
        return optimization_df.merge(
            processed_df[["pid", *variables_of_interest]],
            left_on=[
                "trace_pid",
            ],
            right_on=["pid"],
            how="left",
        )

    def query_optimization_data(
        self,
        prior: str = None,
        include_null: bool = None,
    ) -> pd.DataFrame:
        if prior is None:
            prior = self.prior
        if include_null is None:
            include_null = self.include_null

        subset = self.optimization_data[
            (self.optimization_data["applied_policy"] == "SoftmaxPolicy")
            & (self.optimization_data["prior"] == prior)
        ].copy(deep=True)
        if include_null:
            # random policy doesn't have prior
            subset = pd.concat(
                [
                    subset,
                    self.optimization_data[
                        (self.optimization_data["applied_policy"] == "RandomPolicy")
                    ].copy(deep=True),
                ]
            )

        return subset