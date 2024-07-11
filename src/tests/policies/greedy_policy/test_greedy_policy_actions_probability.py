import gymnasium as gym
from lib.models.Action import Action
from lib.policies.GreedyPolicy import GreedyPolicy


def test_greedy_policy_next_action_mask():
    env = gym.make("Taxi-v3")
    policy = GreedyPolicy(game_env=env, seed=42)  # type: ignore
    policy.game_env.back_to(366)
    result = policy.possible_actions()
    expected = [
        Action.SOUTH,
        Action.NORTH,
        Action.EAST,
        Action.WEST
    ]
    assert (result == exp for exp in expected)
