import gymnasium as gym
from lib.models.Action import ActionProbabilities
from lib.policies.RandomSamplePolicy import RandomSamplePolicy


def test_random_sample_policy_actions_probability():
    env = gym.make("Taxi-v3")
    policy = RandomSamplePolicy(game_env=env)  # type: ignore
    result = policy.actions_probability()
    expected = ActionProbabilities(**{
        "SOUTH": 1/6,
        "NORTH": 1/6,
        "EAST": 1/6,
        "WEST": 1/6,
        "PICK_UP": 1/6,
        "DROP_OFF": 1/6
    })
    assert result == expected
