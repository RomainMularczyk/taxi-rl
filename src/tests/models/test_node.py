import numpy as np
from lib.models.Node import Node
from lib.models.EnvironmentInfo import EnvironmentInfo
from lib.models.Action import Action



def test_node_env_info_actions():
    node: Node = Node(
        action=Action.EAST,
        parent=None,
        depth=2
    )
    env_info = EnvironmentInfo(**{
        "prob": 1.0,
        "action_mask": np.array([0, 1, 1, 0, 1, 0], dtype="int8")  # type: ignore
    })
    node.env_info = env_info

    assert (node.env_info.action_mask == env_info.action_mask).all()

    expected = (Action.NORTH, Action.EAST, Action.PICK_UP)
    result = Action.legal_actions(node.env_info)
    assert result == expected
