import numpy as np
import gymnasium as gym
from lib.formulas.q_learning import QLearning
from lib.policies.RandomSamplePolicy import RandomSamplePolicy


def test_dice_expected_value():
    env = gym.make("Taxi-v3")
    policy = RandomSamplePolicy(env=env)  # type: ignore
    q_learning = QLearning(
        cutoff_score=0,
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n,  # type: ignore
        policy=policy
    )
    result = q_learning.expected_value(
        np.array([1, 2, 3, 4, 5, 6]), np.full((6), 1/6)
    )
    expected = 3.5
    assert result == expected


def test_one_expected_value():
    env = gym.make("Taxi-v3")
    policy = RandomSamplePolicy(env=env)  # type: ignore
    q_learning = QLearning(
        cutoff_score=0,
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n,  # type: ignore
        policy=policy
    )
    result = q_learning.expected_value(
        np.array(10), np.full((1), 1)
    )
    expected = 10.0
    assert result == expected
