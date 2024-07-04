import math
import gymnasium as gym
import numpy as np
from lib.models.Action import Action
from lib.data.q_table import QTable
from lib.policies.LegalSamplePolicy import LegalSamplePolicy
from lib.policies.RandomSamplePolicy import RandomSamplePolicy
from lib.formulas.q_learning import QLearning


def test_random_sample_policy_q_function():
    env = gym.make("Taxi-v3")
    policy = RandomSamplePolicy(env=env, seed=42)  # type: ignore
    q_learning = QLearning(
        cutoff_score=0,
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n,  # type: ignore
        policy=policy
    )
    result = q_learning.q(Action.SOUTH)
    expected = -1.0
    assert result == expected


def test_random_sample_policy_q_function_valid_drop():
    env = gym.make("Taxi-v3")
    policy = RandomSamplePolicy(env=env, seed=42)  # type: ignore
    q_learning = QLearning(
        cutoff_score=0,
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n,  # type: ignore
        policy=policy
    )
    q_learning.policy.env.back_to(479)
    result = q_learning.q(Action.DROP_OFF)
    expected = 20.0
    assert result == expected


def test_random_sample_policy_q_function_invalid_drop():
    env = gym.make("Taxi-v3")
    policy = RandomSamplePolicy(env=env, seed=42)  # type: ignore
    q_learning = QLearning(
        cutoff_score=0,
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n,  # type: ignore
        policy=policy
    )
    q_learning.policy.env.back_to(10)
    result = q_learning.q(Action.DROP_OFF)
    expected = -10.0
    assert result == expected


def test_random_sample_policy_q_star_function():
    env = gym.make("Taxi-v3")
    policy = RandomSamplePolicy(env=env, seed=42)  # type: ignore
    q_learning = QLearning(
        cutoff_score=0,
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n,  # type: ignore
        policy=policy
    )
    result = q_learning.q_star(np.array([1, 0, 0, 0, 0, 0], dtype="int8"))
    expected = (Action.SOUTH, -1.0, -1.0)
    assert result == expected


def test_legal_sample_policy_q_function():
    env = gym.make("Taxi-v3")
    policy = LegalSamplePolicy(env=env, seed=42)  # type: ignore
    q_learning = QLearning(
        cutoff_score=0,
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n,  # type: ignore
        policy=policy
    )
    result = q_learning.q(Action.SOUTH)
    expected = -1.0
    assert result == expected


def test_legal_sample_policy_q_star_function():
    env = gym.make("Taxi-v3")
    policy = LegalSamplePolicy(env=env, seed=42)  # type: ignore
    q_learning = QLearning(
        cutoff_score=0,
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n,  # type: ignore
        policy=policy
    )
    result = q_learning.q_star(np.array([1, 0, 0, 0, 0, 0], dtype="int8"))
    expected = (Action.SOUTH, -1.0, -1.0)
    assert result == expected


def test_random_sample_policy_q_star_function_with_q_table():
    env = gym.make("Taxi-v3")
    policy = RandomSamplePolicy(env=env, seed=42)  # type: ignore
    q_learning = QLearning(
        cutoff_score=0,
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n,  # type: ignore
        policy=policy,
        gamma=0.9,
        data=QTable(
            observation_space=500,
            action_space=6,
            data=np.full((500, 6), 2.0)
        )
    )
    result = q_learning.q_star(np.array([1, 0, 0, 0, 0, 0], dtype="int8"))
    expected_action = Action.SOUTH
    expected_result = 0.8
    assert result[0] == expected_action
    assert float(np.round(result[2], 1)) == expected_result


def test_random_sample_policy_q_function_with_q_table():
    env = gym.make("Taxi-v3")
    policy = RandomSamplePolicy(env=env, seed=42)  # type: ignore
    q_learning = QLearning(
        cutoff_score=0,
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n,  # type: ignore
        policy=policy,
        gamma=0.9,
        data=QTable(
            observation_space=500,
            action_space=6,
            data=np.full((500, 6), 2.0)
        )
    )
    result = q_learning.q(Action.DROP_OFF)
    expected = -8.2
    assert result == expected
