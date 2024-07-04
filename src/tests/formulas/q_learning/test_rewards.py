import gymnasium as gym
from lib.models.Action import Action, ActionWithReward
from lib.policies.LegalSamplePolicy import LegalSamplePolicy
from lib.policies.RandomSamplePolicy import RandomSamplePolicy
from lib.formulas.q_learning import QLearning


def test_rewards_random_sample_policy():
    env = gym.make("Taxi-v3")
    policy = RandomSamplePolicy(env=env, seed=42)  # type: ignore
    q_learning = QLearning(
        cutoff_score=0,
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n,  # type: ignore
        policy=policy
    )
    expected = [
        ActionWithReward(
            action=Action.SOUTH,
            reward=-1.0,
            probability=float(1/6)
        ),
        ActionWithReward(
            action=Action.NORTH,
            reward=-1.0,
            probability=float(1/6)
        ),
        ActionWithReward(
            action=Action.EAST,
            reward=-1.0,
            probability=float(1/6)
        ),
        ActionWithReward(
            action=Action.WEST,
            reward=-1.0,
            probability=float(1/6)
        ),
        ActionWithReward(
            action=Action.PICK_UP,
            reward=-10.0,
            probability=float(1/6)
        ),
        ActionWithReward(
            action=Action.DROP_OFF,
            reward=-10.0,
            probability=float(1/6)
        )
    ]
    result = q_learning.rewards()
    assert result == expected


def test_rewards_legal_sample_policy():
    env = gym.make("Taxi-v3")
    policy = LegalSamplePolicy(env=env, seed=42)  # type: ignore
    q_learning = QLearning(
        cutoff_score=0,
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n,  # type: ignore
        policy=policy
    )
    expected = [
        ActionWithReward(
            action=Action.SOUTH,
            reward=-1.0,
            probability=float(1/3)
        ),
        ActionWithReward(
            action=Action.NORTH,
            reward=-1.0,
            probability=float(1/3)
        ),
        ActionWithReward(
            action=Action.WEST,
            reward=-1.0,
            probability=float(1/3)
        ),
    ]
    result = q_learning.rewards()
    assert result == expected
