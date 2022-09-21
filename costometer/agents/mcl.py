from typing import Any, Callable, Dict, List, Tuple, Union

import hyperopt.pyll.stochastic as hp_dist  # noqa
import numpy as np
from mcl_toolbox.env.generic_mouselab import FeatureMouselabPipelineEnv
from mcl_toolbox.mcrl_modelling.optimizer import get_space
from mcl_toolbox.mcrl_modelling.optimizer import models as model_classes
from mcl_toolbox.models.base_learner import Learner
from mcl_toolbox.utils.experiment_utils import Participant
from mcl_toolbox.utils.fitting_utils import construct_model
from mcl_toolbox.utils.learning_utils import (
    construct_repeated_pipeline,
    create_mcrl_reward_distribution,
    get_normalized_features,
)
from mcl_toolbox.utils.participant_utils import ParticipantIterator
from mouselab.env_utils import get_num_actions
from mouselab.envs.registry import registry

from costometer.agents.vanilla import SymmetricMouselabParticipant


class IRLParticipantIterator(ParticipantIterator):
    """Had to make my own since we don't want self.modify_scores to be called"""

    def __init__(self, participant, click_cost=1):
        self.participant = participant
        self.click_cost = click_cost
        self.clicks = self.participant.clicks
        self.envs = self.participant.envs
        self.rewards = self.participant.scores
        self.taken_paths = self.participant.paths
        self.strategies = self.participant.strategies
        self.temperature = self.participant.temperature
        self.current_trial = 0
        self.current_click = 0


class SymmetricMCLParticipant(SymmetricMouselabParticipant):
    def __init__(
        self,
        experiment_setting: str = "high_increasing",
        reward_dist: str = "categorical",
        model_row: int = None,
        model_attributes: Dict[Any, Any] = None,
        num_trials: int = None,
        ground_truths=None,  # TODO
        trial_ids=None,  # TODO
        cost_function: Callable = None,
        cost_kwargs: Dict[str, Any] = {},
        trace: Dict[str, List] = None,
        policy_function: Callable = None,
        policy_kwargs: Dict[str, Any] = None,
        features: List[str] = None,
        normalized_features: Tuple[Dict[str, Any], Dict[str, Any]] = None,
        params=None,  # TODO
        **kwargs,
    ):
        """MCL Learning participant that only works with symmetric environments for now.  # noqa

        :param experiment_setting: see :py:class:`costometer.agents.vanilla.SymmetricMouselabParticipant`
        :param reward_dist: must be categorical, only here to prevent hardcoding
        :param model_row: MCL model number of row (see rl_models.csv in mcl_toolbox,
         optional either model_attributes or model_row must be provided)
        :param model_attributes: see the documentation on what must be supplied in :py:function:
        :param num_trials: see :py:class:`costometer.agents.vanilla.Participant`
        :param ground_truths: see :py:class:`costometer.agents.vanilla.SymmetricMouselabParticipant`
        :param trial_ids: see :py:class:`costometer.agents.vanilla.SymmetricMouselabParticipant`
        :param cost_function: see :py:class:`costometer.agents.vanilla.Participant`
        :param cost_kwargs: see :py:class:`costometer.agents.vanilla.Participant`
        :param trace: see :py:class:`costometer.agents.vanilla.Participant`
        :param policy_function: see :py:class:`costometer.agents.vanilla.Participant`
        :param policy_kwargs: see :py:class:`costometer.agents.vanilla.Participant`
        :param features: list of features to use, as strings
        :param normalized_features: list of two dictionaries which provide the min and max to normalize features, respectively
        :param params: #TODO
        :return SymmetricMCLParticipant object
        """
        # init form superclass
        super().__init__(
            experiment_setting=experiment_setting,
            num_trials=num_trials,
            ground_truths=ground_truths,
            trial_ids=trial_ids,
            cost_function=cost_function,
            cost_kwargs=cost_kwargs,
            trace=trace,
            policy_function=policy_function,
            policy_kwargs=policy_kwargs if policy_kwargs else {},
            **kwargs,
        )

        # initialize MCL parts
        if reward_dist != "categorical":
            raise ValueError(
                "Code currently does not allow non-categorical distributions, try discretizing."  # noqa
            )
        else:
            reward_distributions = create_mcrl_reward_distribution(experiment_setting)

        self.num_trials = num_trials if num_trials else len(ground_truths)

        # build repeated pipeline
        self.repeated_pipeline = construct_repeated_pipeline(
            registry(experiment_setting).branching,
            reward_distributions,
            self.num_trials,
        )
        self.experiment_setting = experiment_setting

        self.num_actions = get_num_actions(
            registry(self.experiment_setting).branching
        )  # Find out number of actions

        self.features = features

        if model_attributes is None:
            model_attributes = {}

        # only one input is accepted, so XORs
        if (normalized_features is not None) ^ (
            "experiment_name" not in model_attributes
        ):
            raise ValueError(
                "Either 1) experiment name must be provided in "
                "model attributes dictionary"
                " for normalized feature loading"
                " or "
                "2) normalized features must be provided (not both)"
            )
        elif normalized_features is not None:
            self.normalized_features = normalized_features
        elif "experiment_name" in model_attributes:
            self.normalized_features = get_normalized_features(
                model_attributes["experiment_name"]
            )
            del model_attributes["experiment_name"]
        else:
            raise ValueError("No normalized features provided.")

        self.params = params

        # get learner attributes, only one value
        if (model_row is not None) and (len(model_attributes) > 1):
            raise ValueError(
                "Either model row or model attributes can be supplied (not both)!"
            )
        elif model_row:
            (
                self.learner,
                self.learner_attributes,
            ) = self.__build_learner_attributes_from_model_row(model_row)
        elif len(model_attributes) > 1:
            (
                self.learner,
                self.learner_attributes,
            ) = self.__build_learner_attributes_from_dictionary(model_attributes)
        else:
            raise ValueError("One of model row or model attributes must be supplied!")

        # TODO
        self.attach_learner()

        # check if normalized features are missing some features user cares about
        # -> let the user know we need a different input
        if not (
            set(self.features).issubset(set(self.normalized_features[0].keys()))
            and set(self.features).issubset(set(self.normalized_features[1].keys()))
        ):
            raise ValueError(
                "Not all features in normalized features file in mcl_toolbox, "
                "are some of the features new from base MCRL? "
                "Then consider adding normalized_features when "
                "instantiating SymmetricMCLParticipant."
            )

        if self.learner not in model_classes.keys():
            raise ValueError(
                "Learner appears to not be in MCRL toolbox "
                "(or at least is not supported for optimization.)"
            )

        # if cost, specified, deal with before envs
        if cost_function:
            cost = cost_function(**cost_kwargs)
        else:
            cost = -1

        self.mcl_env = FeatureMouselabPipelineEnv(
            self.num_trials,
            pipeline=self.repeated_pipeline,
            ground_truth=ground_truths,
            cost=cost,
            mdp_graph=self.envs[0].mdp_graph,
        )

        self.envs = None

        # changes based on Oct 2021 mcl_toolbox restructuring
        self.mcl_env.attach_features(
            self.learner_attributes["features"],
            self.learner_attributes["normalized_features"],
        )

    def __build_learner_attributes_from_dictionary(
        self, model_attributes: Dict[Any, Any]
    ) -> Tuple[str, Dict[Any, Any]]:
        """# noqa

        :param model_attributes:
        :return: TODO: table
        """
        learner = model_attributes["model"]

        # if features are in model attributes, set self.features
        if ("features" in model_attributes) and (self.features is not None):
            raise ValueError(
                "Expecting either features in model attributes YAML or "
                "as argument to SymmetricMCLParticipant, not both"
            )
        elif ("features" in model_attributes) and (self.features is None):
            self.features = model_attributes["features"]

        model_attributes = {
            "num_actions": self.num_actions,
            "normalized_features": tuple(
                [
                    {
                        normalized_key: normalized_val
                        for normalized_key, normalized_val in normalized_bound.items()
                        if normalized_key in self.features
                    }
                    for normalized_bound in self.normalized_features
                ]
            ),
            "features": self.features,
            **model_attributes,
        }
        return learner, model_attributes

    def __build_learner_attributes_from_model_row(
        self, model_row: int
    ) -> Tuple[str, Dict[Any, Any]]:
        """
        Refactored to use mcl_toolbox.util.fitting_utils rather than duplicate code
            builds the learner attribute dictionary needed to create the LVOC model
        """
        learner, learner_attributes = construct_model(
            model_row, self.num_actions, self.normalized_features
        )

        # construct_model uses all features -- this is used in case we need to subset
        if self.features:
            learner_attributes["features"] = self.features
            learner_attributes["normalized_features"] = self.normalized_features[
                self.features
            ]
        else:
            self.features = learner_attributes["features"]

        return learner, learner_attributes

    def attach_learner(self) -> Learner:
        # sample params not provided
        self.params = self._prior_to_samples(self.learner, self.learner_attributes)

        self.model = model_classes[self.learner](self.params, self.learner_attributes)

        return self.model

    # TODO is this return true
    def _sample_from_weight_prior(
        self, hyperopt_apply: Callable, rng: Union[Callable, int] = np.random
    ) -> List[float]:
        """
        input:
        hyperopt_apply: object of type hyperopt.pyll.base.Apply
        returns:
        float to use as feature weight
        """
        dist = hyperopt_apply.pos_args[0].pos_args[1].name
        args = [
            pos_arg.eval()
            for pos_arg in hyperopt_apply.pos_args[0].pos_args[1].pos_args
        ]

        return eval("hp_dist.{}".format(dist))(*args, rng=rng)

    def _prior_to_samples(
        self, learner: str, learner_attributes: Dict[Any, Any]
    ) -> Dict[Any, Any]:
        """
        input:
        learner: string that identifies the learner (e.g. 'lvoc')
        learner_attributes: dictionary that contains keys
                        that correspond to columns in the file rl_models.csv
        output:
        list of weights
        """
        prior_space = get_space(learner, learner_attributes, "hyperopt")
        params = self.params if self.params else {}

        # if priors are already in params, we don't have to
        # sample each of them
        if "priors" in params:
            prior_space = {
                k: prior_space[k] for k in prior_space.keys() if "prior_" not in k
            }
        else:
            # otherwise, initialize object to collect them in
            samples = {}

        for k, v in prior_space.items():
            # normal prior, only sample if not already in params
            if ("prior_" not in k) and (k not in params):
                params[k] = self._sample_from_weight_prior(v)
            elif "prior_" in k:
                samples[k] = self._sample_from_weight_prior(v)

        if "priors" not in params:
            params["priors"] = np.asarray(
                [
                    samples["prior_{}".format(feature_num)]
                    for feature_num in range(len(learner_attributes["features"]))
                ]
            )

        return params

    def simulate_trajectory(self, force: bool = False) -> List[Dict[str, List]]:
        """
        Sample trajectory from self.model
        """
        trace = self.model.simulate(self.mcl_env)

        trace["ground_truth"] = self.ground_truths
        assert trace["ground_truth"] == trace["envs"] or all(
            el is None for el in trace["ground_truth"]
        )
        del trace["envs"]
        trace["trial_id"] = self.trial_ids

        trace["i_episode"] = list(range(len(trace["a"])))

        # adjust not list trace values
        for k, v in trace.items():
            if not isinstance(v, list):
                trace[k] = [v] * self.num_trials

        # rename a and r to match participant trajectories
        trace["actions"] = trace.pop("a")
        trace["return"] = trace.pop("r")

        if self.trace is not None and not force:
            raise ValueError(
                "Trace already attached, if you need to attach this set force = True"
            )
        else:
            self.trace = trace

        return trace

    def compute_likelihood(self, trace: Dict[str, List]) -> List[float]:
        """Get likelihood of trace given env.model"""
        # get likelihoods

        self.model.simulate(
            self.mcl_env,
            compute_likelihood=True,
            participant=self.__trace_to_participant_iterator(trace),
        )

        # re-nest log likelihoods
        len_episode = [len(actions) for actions in trace["actions"]]
        restructured_likelihood = []
        for episode_num in range(len(trace["actions"]) - 1):
            restructured_likelihood.append(
                self.model.action_log_probs[
                    sum(len_episode[:episode_num]) : sum(len_episode[: episode_num + 1])
                ]
            )
        restructured_likelihood.append(
            self.model.action_log_probs[sum(len_episode[: episode_num + 1]) :]
        )

        # check flattened re-nested list is same as original log likelihoods
        assert [
            item for sublist in restructured_likelihood for item in sublist
        ] == self.model.action_log_probs

        return restructured_likelihood

    def __trace_to_participant_iterator(
        self,
        trial_data: Dict[str, List],
        excluded_trials: List[int] = None,
        pid: int = 0,
        envs: list = None,
        strategies: list = None,
        temperature: Union[int, float] = None,
    ):
        """
        looking at mcl_toolbox.utils.experiment_utils.Participant's method attach_utils,
        we need to attach:
        excluded trials, clicks, paths, envs, scores but not pid, queries, state_rewards

        :param trial_data:
        :param excluded_trials:
        :param pid:
        :param envs:
        :param strategies:
        :param temperature:
        :return:
        """
        participant = Participant(pid)

        # clicks are "a" in mcl outputs or "actions" in mine
        if "a" in trial_data:
            participant.clicks = trial_data["a"]
        elif "actions" in trial_data:
            # also in this case final action needs to be converted to 0
            participant.clicks = [
                [
                    0 if action == self.num_actions else action
                    for action in trial_actions
                ]
                for trial_actions in trial_data["actions"]
            ]
        else:
            participant.clicks = []

        # paths
        if "taken_paths" in trial_data:
            participant.paths = trial_data["taken_paths"]
        else:
            participant.paths = []

        # envs
        if ("envs" in trial_data) and (envs is not None):
            raise ValueError(
                "Two versions of envs provided -- do you need to "
                "pass it to trace_to_participant_iterator?"
            )
        elif "envs" in trial_data:
            participant.envs = trial_data["envs"]
        elif envs is not None:
            participant.envs = envs
        else:
            participant.envs = []

        # rewards/costs for each planning action
        if "costs" in trial_data:
            participant.scores = trial_data["costs"]
        elif "rewards" in trial_data:
            participant.scores = trial_data["rewards"]
        else:
            participant.scores = []

        # excluded trials
        if excluded_trials is not None:
            participant.excluded_trials = excluded_trials
            participant.exclude_trial_data()

        if strategies is None:
            participant.strategies = []
        else:
            participant.strategies = strategies

        if temperature is None:
            participant.temperature = 1
        else:
            participant.temperature = temperature

        # s if from mcl_toolbox, states otherwise
        if "s" in trial_data:
            participant.strategies = trial_data["s"]
        if "states" in trial_data:
            participant.strategies = trial_data["states"]

        # w is only found in mcl_toolbox
        if "w" in trial_data:
            participant.weights = trial_data["w"]

        # r if from mcl_toolbox, return if other participant
        if "r" in trial_data:
            participant.returns = trial_data["r"]
        elif "return" in trial_data:
            participant.returns = trial_data["return"]

        return IRLParticipantIterator(participant)
