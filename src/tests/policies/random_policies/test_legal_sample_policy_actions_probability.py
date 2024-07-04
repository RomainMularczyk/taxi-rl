import gymnasium as gym
from lib.models.Action import ActionProbabilities
from lib.policies.LegalSamplePolicy import LegalSamplePolicy


def test_random_sample_policy_three_legal_actions_probabilities():
    env = gym.make("Taxi-v3")
    policy = LegalSamplePolicy(env=env, seed=42)  # type: ignore
    result = policy.actions_probability()
    expected = ActionProbabilities(**{
        "SOUTH": 1/3,
        "NORTH": 1/3,
        "WEST": 1/3
    })
    assert result == expected


def test_random_sample_policy_with_pick_up_legal_action_probabilities():
    env = gym.make("Taxi-v3")
    policy = LegalSamplePolicy(env=env, seed=10)  # type: ignore
    result = policy.actions_probability()
    expected = ActionProbabilities(**{
        "NORTH": 1/3,
        "EAST": 1/3,
        "PICK_UP": 1/3
    })
    assert result == expected


def test_random_sample_policy_two_legal_actions_probabilities():
    env = gym.make("Taxi-v3")
    policy = LegalSamplePolicy(env=env, seed=25)  # type: ignore
    result = policy.actions_probability()
    expected = ActionProbabilities(**{
        "SOUTH": 1/2,
        "WEST": 1/2
    })
    assert result == expected


def test_random_sample_policy_four_legal_actions_probabilities():
    env = gym.make("Taxi-v3")
    policy = LegalSamplePolicy(env=env, seed=39)  # type: ignore
    result = policy.actions_probability()
    expected = ActionProbabilities(**{
        "SOUTH": 1/4,
        "NORTH": 1/4,
        "EAST": 1/4,
        "WEST": 1/4
    })
    assert result == expected
