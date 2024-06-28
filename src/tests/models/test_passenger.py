import gymnasium as gym
from lib.environment.environment import GameEnvironment
from lib.models.Passenger import Passenger


def test_passenger_location():
    env = gym.make("Taxi-v3")
    env.reset(seed=42)
    env = GameEnvironment(env=env)  # type: ignore
    result = Passenger.location(env.state)
    expected = Passenger(3)
    assert result == expected


def test_passenger_absolute_location():
    env = gym.make("Taxi-v3")
    env.reset(seed=42)
    env = GameEnvironment(env=env)  # type: ignore
    result = Passenger.location(env.state, absolute=True)
    expected = (4, 3)
    assert result == expected
