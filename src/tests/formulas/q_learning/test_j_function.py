import gymnasium as gym
import numpy as np
from lib.policies.LegalSamplePolicy import LegalSamplePolicy
from lib.policies.RandomSamplePolicy import RandomSamplePolicy
from lib.formulas.q_learning import QLearning


def test_j_random_sample_policy():
    env = gym.make("Taxi-v3")
    policy = RandomSamplePolicy(env=env, seed=42)  # type: ignore
    q_learning = QLearning(
        cutoff_score=0,
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n,  # type: ignore
        policy=policy
    )
    q_values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    probs = np.full(6, float(1/6))
    expected = (q_values * probs).sum()
    result = q_learning.j()
    assert result == expected


def test_j_legal_sample_policy():
    env = gym.make("Taxi-v3")
    policy = LegalSamplePolicy(env=env, seed=42)  # type: ignore
    q_learning = QLearning(
        cutoff_score=0,
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n,  # type: ignore
        policy=policy
    )
    q_values = np.array([0.0, 0.0, 0.0])
    probs = np.full(3, float(1/3))
    expected = (q_values * probs).sum()
    result = q_learning.j()
    assert result == expected
