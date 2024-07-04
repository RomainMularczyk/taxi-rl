import gymnasium as gym
from lib.models.Action import Action, ActionWithReward
from lib.policies.MaxPolicy import MaxPolicy


def test_max_policy_next_action():
    env = gym.make("Taxi-v3")
    policy = MaxPolicy(env=env, seed=42)  # type: ignore
    result = policy.next_action()
    expected = ActionWithReward(
        action=Action.SOUTH,
        reward=-1.0
    )
    assert result == expected
