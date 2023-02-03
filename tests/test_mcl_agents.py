import json
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml
from mcl_toolbox.models.lvoc_models import LVOC
from mouselab.envs.registry import register
from mouselab.envs.reward_settings import high_decreasing_reward, high_increasing_reward
from parameterized import parameterized

from costometer.agents.mcl import SymmetricMCLParticipant

register(
    name="small_increasing",
    branching=[2, 2],
    reward_inputs="depth",
    reward_dictionary=high_increasing_reward,
)
register(
    name="small_decreasing",
    branching=[2, 2],
    reward_inputs="depth",
    reward_dictionary=high_decreasing_reward,
)

# start creating test cases
test_cases_model = [
    [
        "high_increasing",
        {
            "num_trials": 5,
            "model_row": 1729,
            "model_attributes": {"experiment_name": "F1"},
        },
        LVOC,
    ]
]

mcl_instantiation_test_cases = [
    [model_row, "312_2_4_24", "F1", 5] for model_row in [1729, 1825, 1921]
]


class TestMCL(unittest.TestCase):
    @parameterized.expand(deepcopy(test_cases_model))
    def test_instantiate_participant(self, setting, additional_settings, model_class):
        participant = SymmetricMCLParticipant(setting, **additional_settings)
        self.assertIsInstance(participant, SymmetricMCLParticipant)

    @parameterized.expand(deepcopy(test_cases_model))
    def test_sample_learner(self, setting, additional_settings, model_class):
        participant = SymmetricMCLParticipant(setting, **additional_settings)
        participant.sample_learner()
        self.assertIsInstance(participant.model, model_class)

    @parameterized.expand(deepcopy(test_cases_model))
    def test_simulate_trajectory(self, setting, additional_settings, model_class):
        participant = SymmetricMCLParticipant(setting, **additional_settings)
        participant.sample_learner()
        trajectory = participant.simulate_trajectory()
        self.assertIsInstance(trajectory, dict)

    @parameterized.expand(deepcopy(test_cases_model))
    def test_compute_likelihood(self, setting, additional_settings, model_class):
        participant = SymmetricMCLParticipant(setting, **additional_settings)
        participant.sample_learner()
        trajectory = participant.simulate_trajectory()
        participant.compute_likelihood(participant.trace)
        self.assertIsInstance(trajectory, dict)

    @parameterized.expand(mcl_instantiation_test_cases)
    def test_mcl_instantiation_methods(
        self, model_row, ground_truths, experiment_name, num_trials
    ):
        # load yaml inputs
        path_to_yamls = [
            Path(__file__).parents[1].joinpath("tests/inputs/features/habitual.yaml"),
            Path(__file__)
            .parents[1]
            .joinpath(f"tests/inputs/mcl_run/{model_row}.yaml"),
        ]
        args = {}
        for yaml_path in path_to_yamls:
            with open(str(yaml_path), "r") as stream:
                args = {**args, **yaml.safe_load(stream)}

        # get ground truths
        with open(
            Path(__file__)
            .parents[1]
            .joinpath(f"tests/inputs/rewards/{ground_truths}.json")
        ) as json_file:
            ground_truths = json.load(json_file)
        curr_ground_truths = np.random.choice(ground_truths, num_trials)

        ground_truths = [
            ground_truth["stateRewards"] for ground_truth in curr_ground_truths
        ]
        trial_ids = [ground_truth["trial_id"] for ground_truth in curr_ground_truths]

        # create participant_from_row
        participant_from_row = SymmetricMCLParticipant(
            model_row=model_row,
            model_attributes={"experiment_name": experiment_name},
            ground_truths=ground_truths,
            trial_ids=trial_ids,
        )
        participant_from_row.simulate_trajectory()

        # create participant_from_yaml
        participant_from_yaml = SymmetricMCLParticipant(
            model_attributes={"experiment_name": experiment_name, **args},
            params=participant_from_row.params,
            ground_truths=ground_truths,
            trial_ids=trial_ids,
        )
        participant_from_yaml.simulate_trajectory()

        self.assertEqual(
            participant_from_row.compute_likelihood(trace=participant_from_row.trace),
            participant_from_yaml.compute_likelihood(trace=participant_from_row.trace),
        )
        self.assertEqual(
            participant_from_row.compute_likelihood(trace=participant_from_yaml.trace),
            participant_from_yaml.compute_likelihood(trace=participant_from_yaml.trace),
        )

        # self.assertTrue(np.all(np.exp(participant_from_row.compute_likelihood(trace=participant_from_row.trace)) - np.exp(  # noqa
        #     participant_from_yaml.compute_likelihood(trace=participant_from_row.trace)) < 10e-7))  # noqa
