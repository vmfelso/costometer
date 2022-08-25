from copy import deepcopy
from typing import Dict, List, Type

import numpy as np

from costometer.agents.vanilla import Participant
from costometer.inference.base import BaseInference


class KubalaInference(BaseInference):
    def __init__(
        self, traces: List[Dict[str, List]], sampled_agents: List[Type[Participant]]
    ):
        """
        This class sets things up for the \
        Kubala Bayesian Inverse Reinforcement Learning Algorithm:
        Kubala, Konidaris, Greenwald. Inverse Reinforcement Learning
        from a Learning Agent. RLDM 2019.

        Steps:

        1. Initiation phase: sample learners, and compute initial policies \
        (this constructor)
        2. Update phase: given new (s,a) compute normalized probability \
        of particle given sequence of (s,a) \
        (this constructor, \
         via :py:meth:`costometer.inference.kubala.Kubala.update_sample_probability`)
        3. Query phase: return reward function parameters \
        (can be done with help of \
         :py:meth:`costometer.inference.mle.BaseInference.get_best_agents`)

        :ivar sample_probability: dictionary of {trace_index : \
        np.array of shape (num_actions + 1, num_agents)}. \
        Entry (i,j) in the matrix under trajectory index k should be \
        the probability that agent j is the best fitting agent \
        for trajectory k when considering all actions up to (but not including) \
        the ith action (0-indexed)
        :ivar sample_probability_prior: provides prior probability \
        (used on 0th row of np.arrays,  e.g., probability that agent j is the \
        best-fitting agent for trajectory k when no actions are considered.) \
        Hard-coded to uniform probability (1 / number of agents.)
        :ivar pi: see :py:class:`costometer.inference.mle.BaseInference`
        :param traces: see :py:class:`costometer.inference.mle.BaseInference`
        :param sampled_agents: see :py:class:`costometer.inference.mle.BaseInference`
        """
        super().__init__(traces)

        # save inputs
        self.traces = traces
        if not isinstance(sampled_agents, list):
            raise ValueError("Agents must be a list")
        self.sampled_agents = sampled_agents

        # get num agents and trials
        self.num_agents = len(sampled_agents)
        # flatten trace actions
        self.num_actions = [len(np.concatenate(trace["actions"])) for trace in traces]

        # intialize pi {trace_idx : likelihoods in action x particle}
        self.pi = {
            trace_idx: np.zeros((self.num_actions[trace_idx], self.num_agents))
            for trace_idx, _ in enumerate(traces)
        }

        self.sample_probability = None
        self.sample_probability_prior = {
            trace_idx: (
                np.ones((self.num_actions[trace_idx] + 1, self.num_agents))
                * 1
                / float(self.num_agents)
            )
            for trace_idx, _ in enumerate(traces)
        }

        # get posterior sample probability
        self.update_sample_probability()

    def update_sample_probability(self) -> None:
        """
        This function combines two pieces of information:

        1. The instance variable "pi"
        2. The instance variable "sample_probability_prior"

        This function computes the posterior probability of \
        an agent being the best agent, otherwise known as \
        the instance variable "sample_probability".

        :return: None
        """
        self.sample_probability = deepcopy(self.sample_probability_prior)

        for trace_idx, trace in enumerate(self.traces):
            for action in range(self.num_actions[trace_idx]):
                # pi is in log lik so need to exponentiate it
                agent_probs = np.multiply(
                    np.exp(self.pi[trace_idx][action, :]),
                    self.sample_probability[trace_idx][action, :],
                )
                # new sample probabilities normalized agent probs
                self.sample_probability[trace_idx][action + 1, :] = (
                    agent_probs / agent_probs.sum()
                )

    def get_best_agents(self) -> List[Type[Participant]]:
        """
        :return: outputs list of best learners for each trace"""
        # return best agent in final trial
        best_agent_indices = [
            np.argmax(trace_sample_prob[-1, :])
            for trace_idx, trace_sample_prob in self.sample_probability.items()
        ]

        return [
            self.sampled_agents[best_agent_index]
            for best_agent_index in best_agent_indices
        ]
