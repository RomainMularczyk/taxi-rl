import numpy as np
from lib.models.EnvironmentInfo import EnvironmentInfo
from lib.models.Action import Action


def test_legal_actions():
    env_info = EnvironmentInfo(**{
        "prob": 1.0,
        "action_mask": np.array([0, 0, 1, 1, 0, 1])  # type: ignore
    })
    result = Action.legal_actions(env_info)
    expected = (Action.EAST, Action.WEST, Action.DROP_OFF)
    assert result == expected


def test_no_legal_actions():
    env_info = EnvironmentInfo(**{
        "prob": 1.0,
        "action_mask": np.array([0, 0, 0, 0, 0, 0])  # type: ignore
    })
    result = Action.legal_actions(env_info)
    expected = ()
    assert result == expected


def test_all_legal_actions():
    env_info = EnvironmentInfo(**{
        "prob": 1.0,
        "action_mask": np.array([1, 1, 1, 1, 1, 1])  # type: ignore
    })
    result = Action.legal_actions(env_info)
    expected = (
        Action.SOUTH,
        Action.NORTH,
        Action.EAST,
        Action.WEST,
        Action.PICK_UP,
        Action.DROP_OFF
    )
    assert result == expected
