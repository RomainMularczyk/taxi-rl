import gymnasium as gym
import pandas as pd
import pytest
from lib.models.Action import Action
from lib.data.q_table import QTable


def test_q_table_size():
    env = gym.make("Taxi-v3")
    q_table = QTable(
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n  # type: ignore
    )
    expected = 500
    result = len(q_table.values)
    assert result == expected


def test_q_table_wrong_values():
    with pytest.raises(ValueError):
        QTable(
            observation_space="500",  # type: ignore
            action_space=list(Action)  # type: ignore
        )


def test_q_table_index():
    env = gym.make("Taxi-v3")
    q_table = QTable(
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n  # type: ignore
    )
    result = q_table[0, 0]
    expected = 0.0
    assert result == expected


def test_q_table_index_row_last_value():
    env = gym.make("Taxi-v3")
    q_table = QTable(
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n  # type: ignore
    )
    result = q_table[499, 0]
    expected = 0.0
    assert result == expected


def test_q_table_index_row_out_of_bound():
    env = gym.make("Taxi-v3")
    q_table = QTable(
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n  # type: ignore
    )
    with pytest.raises(IndexError):
        q_table[500, 0]


def test_q_table_index_col_out_of_bound():
    env = gym.make("Taxi-v3")
    q_table = QTable(
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n  # type: ignore
    )
    with pytest.raises(IndexError):
        q_table[500, 6]


def test_q_table_update():
    env = gym.make("Taxi-v3")
    q_table = QTable(
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n  # type: ignore
    )
    q_table[0, 0] = 10.0
    expected = 10.0
    result = q_table[0, 0]
    assert result == expected


def test_q_table_update_row_out_of_bound():
    env = gym.make("Taxi-v3")
    q_table = QTable(
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n  # type: ignore
    )
    with pytest.raises(IndexError):
        q_table[500, 0] = 10.0


def test_q_table_update_col_out_of_bound():
    env = gym.make("Taxi-v3")
    q_table = QTable(
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n  # type: ignore
    )
    with pytest.raises(IndexError):
        q_table[0, 6] = 10.0


def test_q_table_dataframe_col_index():
    env = gym.make("Taxi-v3")
    q_table = QTable(
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n  # type: ignore
    )
    result = q_table.display_table(scrollable=False).columns  # type: ignore
    expected = pd.Index(
        ["SOUTH", "NORTH", "EAST", "WEST", "PICK_UP", "DROP_OFF"]
    )
    assert (result == expected).all()


def test_q_table_dataframe_row_index():
    env = gym.make("Taxi-v3")
    q_table = QTable(
        observation_space=env.observation_space.n,  # type: ignore
        action_space=env.action_space.n  # type: ignore
    )
    result = q_table.display_table(scrollable=False).index  # type: ignore
    expected = pd.RangeIndex(start=0, stop=500, step=1)
    assert list(result) == list(expected)
