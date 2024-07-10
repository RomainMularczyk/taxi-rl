import numpy as np
import gymnasium as gym
from lib.models.Action import Action, ActionWithReward
from lib.policies.GreedyPolicy import GreedyPolicy


def test_greedy_policy_next_action():
    env = gym.make("Taxi-v3")
    policy = GreedyPolicy(game_env=env, seed=42)  # type: ignore
    policy.game_env.back_to(366)
    result = policy.next_action()
    expected = [
        ActionWithReward(
            action=Action.SOUTH,
            reward=-1.0,
            probability=1.0
        ),
        ActionWithReward(
            action=Action.NORTH,
            reward=-1.0,
            probability=1.0
        ),
        ActionWithReward(
            action=Action.EAST,
            reward=-1.0,
            probability=1.0
        ),
        ActionWithReward(
            action=Action.WEST,
            reward=-1.0,
            probability=1.0
        )
    ]
    assert any(result == exp for exp in expected)


def test_greedy_policy_next_action_mask():
    env = gym.make("Taxi-v3")
    policy = GreedyPolicy(game_env=env, seed=42)  # type: ignore
    policy.game_env.back_to(366)
    result = policy.next_action(
        mask=np.array([1, 0, 0, 0, 0, 0], dtype="int8")
    )
    expected = ActionWithReward(
        action=Action.SOUTH,
        reward=-1.0,
        probability=1.0
    )
    assert result == expected
