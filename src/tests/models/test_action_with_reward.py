import numpy as np
from lib.models.Action import Action, ActionWithReward


def test_flatten_action_with_reward():
    awr = [
        ActionWithReward(
            action=Action.SOUTH,
            reward=-1.0,
            probability=float(1/6)
        ),
        ActionWithReward(
            action=Action.NORTH,
            reward=-1.0,
            probability=float(1/6)
        )
    ]
    expected = np.array([[-1.0, -1.0], [float(1/6), float(1/6)]])
    result = ActionWithReward.flatten(awr)
    assert (result == expected).all()
