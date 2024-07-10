import gymnasium as gym
from lib.models.Action import Action
from lib.policies.LegalSamplePolicy import LegalSamplePolicy
from lib.policies.RandomSamplePolicy import RandomSamplePolicy
from lib.formulas.q_learning import QLearning


def test_random_sample_policy_v_function():
    env = gym.make("Taxi-v3")
    policy = RandomSamplePolicy(game_env=env, seed=42)  # type: ignore
    q_learning = QLearning(
        cutoff_score=0,
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n,  # type: ignore
        policy=policy
    )
    #result = q_learning.v(Action.SOUTH)


def test_random_sample_policy_v_star_function():
    env = gym.make("Taxi-v3")
    policy = RandomSamplePolicy(game_env=env, seed=42)  # type: ignore
    q_learning = QLearning(
        cutoff_score=0,
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n,  # type: ignore
        policy=policy
    )
    #result = q_learning.v_star()


def test_legal_sample_policy_v_function():
    env = gym.make("Taxi-v3")
    policy = LegalSamplePolicy(game_env=env, seed=42)  # type: ignore
    q_learning = QLearning(
        cutoff_score=0,
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n,  # type: ignore
        policy=policy
    )


def test_legal_sample_policy_v_star_function():
    env = gym.make("Taxi-v3")
    policy = LegalSamplePolicy(game_env=env, seed=42)  # type: ignore
    q_learning = QLearning(
        cutoff_score=0,
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n,  # type: ignore
        policy=policy
    )
