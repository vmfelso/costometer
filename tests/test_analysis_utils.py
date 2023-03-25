import pytest


@pytest.fixture(params=["TrialByTrial", "TrialByTrialAll"])
def log_likelihood_test_cases(request):
    raise NotImplementedError
    # # setup
    # analysis_obj = AnalysisObject(request.param)
    # trace_df = traces_to_df(
    #     get_trajectories_from_participant_data(analysis_obj.mouselab_trials)
    # )
    # trial_by_trial_df = analysis_obj.get_trial_by_trial_likelihoods()
    #
    # yield trace_df, trial_by_trial_df


def test_log_likelihood(log_likelihood_test_cases):
    raise NotImplementedError
    # q_dict = load_q_file(
    #     "high_increasing",
    #     cost_function=linear_depth,
    #     cost_params={"depth_cost_weight": 0, "static_cost_weight": 1},
    #     path=Path(__file__).parents[4].joinpath("cluster/data/q_files"),
    # )
    # print(np.exp([q_dict[(row["states"], action)] if action == 13 or
    #       hasattr(row["states"][action], 'sample') else 0
    #               for action in range(1, 14)
    #               ][row["actions"]]))
    # trace_df.apply(lambda row: q_dict[(row["states"], row["actions"])], axis=1)
    #
