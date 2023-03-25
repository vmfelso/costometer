import unittest
from pathlib import Path
from shutil import rmtree

from mouselab.cost_functions import linear_depth
from mouselab.envs.registry import register
from mouselab.envs.reward_settings import high_decreasing_reward, high_increasing_reward
from parameterized import parameterized

from costometer.utils import save_q_values_for_cost

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

rmtree("./outputs/", ignore_errors=True)
output_path = Path(__file__).parents[0].joinpath("./outputs")
output_path.mkdir(parents=True, exist_ok=True)

test_save_q = [
    [
        "small_increasing",
        output_path,
        {
            "cost_function": linear_depth,
            "cost_params": {"depth_cost_weight": 10, "static_cost_weight": 10},
        },
    ],
    [
        "small_decreasing",
        output_path,
        {
            "cost_function": linear_depth,
            "cost_params": {"depth_cost_weight": 0, "static_cost_weight": 1},
        },
    ],
]


class TestUtils(unittest.TestCase):
    @parameterized.expand(test_save_q)
    def test_save_q_values_for_cost(self, experiment_setting, path, cost_kwargs):
        beginning_file_num = len(list(path.glob("*")))
        save_q_values_for_cost(experiment_setting, path=path, **cost_kwargs)
        ending_file_num = len(list(path.glob("*")))
        # just check file is being saved
        self.assertEqual(ending_file_num - beginning_file_num, 1)

    @parameterized.expand(test_save_q)
    def test_human_to_trace(self, experiment_setting, path, cost_kwargs):
        # TODO
        raise NotImplementedError
