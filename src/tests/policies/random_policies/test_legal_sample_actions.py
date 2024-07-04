import gymnasium as gym
from lib.models.Action import Action
from lib.policies.LegalSamplePolicy import LegalSamplePolicy


def test_random_sample_policy_three_legal_actions():
    env = gym.make("Taxi-v3")
    policy = LegalSamplePolicy(env=env, seed=42)  # type: ignore
    result = policy.legal_actions()
    expected = [Action.SOUTH, Action.NORTH, Action.WEST]
    assert result == expected


def test_random_sample_policy_with_pick_up_legal_actions():
    env = gym.make("Taxi-v3")
    policy = LegalSamplePolicy(env=env, seed=10)  # type: ignore
    result = policy.legal_actions()
    expected = [Action.NORTH, Action.EAST, Action.PICK_UP]
    assert result == expected


def test_random_sample_policy_two_legal_actions():
    env = gym.make("Taxi-v3")
    policy = LegalSamplePolicy(env=env, seed=25)  # type: ignore
    result = policy.legal_actions()
    expected = [Action.SOUTH, Action.WEST]
    assert result == expected


def test_random_sample_policy_four_legal_actions():
    env = gym.make("Taxi-v3")
    policy = LegalSamplePolicy(env=env, seed=39)  # type: ignore
    result = policy.legal_actions()
    expected = [Action.SOUTH, Action.NORTH, Action.EAST, Action.WEST]
    assert result == expected
