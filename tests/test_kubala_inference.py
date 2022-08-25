import unittest
from pathlib import Path

import dill as pickle
import numpy as np
from mouselab.cost_functions import linear_depth
from mouselab.envs.registry import register
from mouselab.envs.reward_settings import high_increasing_reward
from mouselab.policies import RandomPolicy, SoftmaxPolicy
from parameterized import parameterized

from costometer.agents.vanilla import SymmetricMouselabParticipant
from costometer.inference.kubala import KubalaInference
from costometer.utils import get_param_string, save_q_values_for_cost

# make three test agents, and test with three traces, one from each
register(
    name="small_increasing",
    branching=[2, 2],
    reward_inputs="depth",
    reward_dictionary=high_increasing_reward,
)

setting = "small_increasing"
num_episodes = 30
cost_inputs = [
    {"depth_cost_weight": 0, "static_cost_weight": 1},
    {"depth_cost_weight": 0, "static_cost_weight": 1},
    {"depth_cost_weight": 10, "static_cost_weight": 10},
]
policy_functions = [RandomPolicy, SoftmaxPolicy, SoftmaxPolicy]
policy_kwargs = [
    {},
    {"preference": {}, "temp": 1, "noise": 0},
    {"preference": {}, "temp": 1, "noise": 0},
]
# create Q values if needed
input_path = Path(__file__).parents[0].joinpath("./inputs")
input_path.mkdir(parents=True, exist_ok=True)
for cost_idx, cost_input in enumerate(cost_inputs):
    parameter_string = get_param_string(cost_input)
    matching_files = list(input_path.glob(f"Q_{setting}_{parameter_string}_*.pickle"))
    if len(matching_files) == 0:
        info = save_q_values_for_cost(
            setting, cost_function=linear_depth, cost_params=cost_input, path=input_path
        )
    else:
        with open(matching_files[0], "rb") as f:
            info = pickle.load(f)

    if policy_functions[cost_idx] != RandomPolicy:
        policy_kwargs[cost_idx]["preference"] = info["q_dictionary"]

sampled_agents = []
for input_idx, cost_input in enumerate(cost_inputs):
    policy_function = policy_functions[input_idx]
    policy_kwarg = policy_kwargs[input_idx]
    agent = SymmetricMouselabParticipant(
        setting,
        num_trials=num_episodes,
        cost_function=linear_depth,
        cost_kwargs=cost_input,
        policy_function=policy_function,
        policy_kwargs=policy_kwarg,
    )
    sampled_agents.append(agent)

# want test cases to:
# - try all sampled agents
# - one for each trace
# - have correct index
kubala_test_cases = []
for sample_idx, sampled_agent in enumerate(sampled_agents):
    # get trace
    trace = sampled_agent.simulate_trajectory()
    kubala_test_cases.append([[trace], sampled_agents, sample_idx])


class TestKubala(unittest.TestCase):
    @parameterized.expand(kubala_test_cases)
    def test_instantiate(self, trace, sampled_agents, sample_idx):
        kubala_algorithm = KubalaInference(trace, sampled_agents)
        self.assertIsInstance(kubala_algorithm, KubalaInference)

    @parameterized.expand(kubala_test_cases)
    def test_get_pi(self, trace, sampled_agents, sample_idx):
        kubala_algorithm = KubalaInference(trace, sampled_agents)
        kubala_algorithm.get_pi()

        # check probabilities between 0 and 1
        probabilities = np.exp(np.vstack(list(kubala_algorithm.pi.values())))
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))

    @parameterized.expand(kubala_test_cases)
    def test_kubala_run(self, trace, sampled_agents, sample_idx):
        kubala_algorithm = KubalaInference(trace, sampled_agents)

        # run algorithm
        kubala_algorithm.run()

        # get best model
        best_model = kubala_algorithm.get_best_agents()

        # test results from get_sample_probability
        probabilities = np.vstack(list(kubala_algorithm.sample_probability.values()))
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))

        # test if correct agent inferred
        self.assertTrue(best_model, [sampled_agents[sample_idx]])
