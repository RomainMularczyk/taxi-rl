from lib.formulas.q_learning import QLearning
from lib.policies.RandomSamplePolicy import RandomSamplePolicy


def test_dice_expected_value():
    policy = RandomSamplePolicy()
    q_learning = QLearning(
        initial_seed=0,
        cutoff_score=0,
        observation_space=5,
        action_space=5,
        policy=policy
    )
    result = q_learning.expected_value(6)
    expected = 3.5
    assert result == expected


def test_one_expected_value():
    pass
