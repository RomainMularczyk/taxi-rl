import math
from lib.models.Action import Action
from lib.models.Node import Node


class MonteCarloTree:
    """
    Representation of a Tree. Each branch represent an action and
    each node represent the position on the map.
    """

    def __init__(self, actions: list[Action], depth: int):
        """
        actions: [Action]
            All the actions the agent can make.
        depth: int
            The maximum allowed amount of steps the agent can make
            if it's the furthest from the pickup point.
        """
        self.depth = depth
        self.actions = actions
        self.root_node = Node(
            action=None,
            state=None,
            parent=None,
            children=[],
            depth=0
        )
        self.pick_best_score = -math.inf
        self.drop_best_score = -math.inf
        # Build the tree
        self._create_tree(
            root_node=self.root_node,
            max_depth=self.depth
        )

    def _create_tree(self, root_node: Node, max_depth: int) -> None:
        """
        Create the Tree starting from the root_node.

        Attributes
        ----------
        root_node: Node
            The node we want to create with its children.
        max_depth: int
            The depth we want our Tree to be.

        Returns
        -------
        Node
            The root node of the Tree.
        """
        for action in self.actions:
            child_node = Node(
                action=action,
                path=f"{root_node.depth+1}{action.to_letter()}",
                state=None,
                parent=root_node,
                depth=root_node.depth + 1,
                children=[]
            )
            if child_node.depth > max_depth:
                return None
            else:
                root_node.children.append(child_node)
                self._create_tree(
                    child_node,
                    max_depth=max_depth
                )
