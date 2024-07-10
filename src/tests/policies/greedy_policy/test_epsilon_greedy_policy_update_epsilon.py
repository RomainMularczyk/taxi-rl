import gymnasium as gym
import numpy as np
from lib.policies.EpsilonGreedyPolicy import EpsilonGreedyPolicy


def test_epsilon_greedy_policy_update_epsilon_step_1():
    env = gym.make("Taxi-v3")
    policy = EpsilonGreedyPolicy(game_env=env, seed=42)  # type: ignore
    policy._steps = 1
    policy._update_epsilon()
    result = policy.epsilon
    expected = 0.9
    assert np.round(result, 2) == expected


def test_epsilon_greedy_policy_update_epsilon_step_10():
    env = gym.make("Taxi-v3")
    policy = EpsilonGreedyPolicy(game_env=env, seed=42)  # type: ignore
    policy._steps = 10
    policy._update_epsilon()
    result = policy.epsilon
    expected = 0.37
    assert np.round(result, 2) == expected


def test_epsilon_greedy_policy_update_epsilon_step_100():
    env = gym.make("Taxi-v3")
    policy = EpsilonGreedyPolicy(game_env=env, seed=42)  # type: ignore
    policy._steps = 100
    policy._update_epsilon()
    result = policy.epsilon
    expected = 0.0
    assert np.round(result, 2) == expected
