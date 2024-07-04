import pytest
import gymnasium as gym
from lib.policies.LegalSamplePolicy import LegalSamplePolicy


def test_back_to_state_10():
    env = gym.make("Taxi-v3")
    policy = LegalSamplePolicy(env=env, seed=42)  # type: ignore
    policy.env.back_to(10)
    result = policy.env.state
    expected = 10
    assert result == expected


def test_back_to_state_1000():
    env = gym.make("Taxi-v3")
    policy = LegalSamplePolicy(env=env, seed=42)  # type: ignore
    with pytest.raises(ValueError, match="contained between 0 and 500"):
        policy.env.back_to(1000)
