import pytest

torch = pytest.importorskip("torch")

from hippotorch.memory.regression import run_replay_regression


def test_regression_harness_reports_improvement():
    torch.manual_seed(0)
    result = run_replay_regression(
        batch_size=12, num_good=6, num_distractor=18, episode_length=3
    )

    assert result.semantic_mean_reward > result.uniform_mean_reward
    assert result.reward_improvement > 0.25
    assert "memory_size" in result.semantic_stats
    assert "memory_size" in result.uniform_stats
