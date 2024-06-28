import gymnasium as gym
from lib.environment.environment import GameEnvironment
from lib.models.Destination import Destination


def test_destination_location():
    env = gym.make("Taxi-v3")
    env.reset(seed=42)
    env = GameEnvironment(env=env)  # type: ignore
    result = Destination.location(env.state)
    expected = Destination(2)
    assert result == expected


def test_destination_absolute_location():
    env = gym.make("Taxi-v3")
    env.reset(seed=42)
    env = GameEnvironment(env=env)  # type: ignore
    result = Destination.location(env.state, absolute=True)
    expected = (4, 0)
    assert result == expected
