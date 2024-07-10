from lib.data.monte_carlo_tree import MonteCarloTree
from lib.models.Action import Action
from lib.models.Node import Node


def test_create_monte_carlo_tree():
    actions = [a for a in Action]
    depth = 2
    result = MonteCarloTree(actions=actions, depth=depth)

    expected_root_node = Node(
        action=None,
        parent=None,
        children=[],
        state=None,
        depth=0
    )
    expected_root_node_children = [
        Node(
            depth=expected_root_node.depth + 1,
            action=action,
            parent=expected_root_node,
            children=None,
            state=None
        ) for action in actions
    ]

    assert result.root_node.path == expected_root_node.path
    for child_node in result.root_node.children:
       assert child_node.parent.path == expected_root_node.path
