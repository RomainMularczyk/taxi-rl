from __future__ import annotations
import textwrap
from typing import List
from lib.models.Action import Action


class Node:
    """
    This class represent a Node in a Tree.

    Attributes
    ----------
    action: Action
        The action associated with this Node.
    path: str
        The path name of this Node. Number is the layer level and letter is the action.
    state: int
        The state of this Node depending on the GameEnvironment.
    parent: Node
        The parent Node.
    children: List[Node]
        The children of the Node.
    """
    def __init__(
        self, 
        action: Action | None, 
        state: int | None, 
        depth: int,
        parent: Node | None, 
        children: List[Node] | List = [],
        path: str | None = '',
    ):
        self.action = action
        self.path = path
        self.depth = depth
        self.state = state
        self.parent = parent
        self.children = children


    def __repr__(self):
        try:
            return textwrap.dedent(f"""
                [Node] {self.path}
                [Parent] {self.parent.path}
                [Children] {[c.path for c in self.children]}
            """)
        except AttributeError:
            return textwrap.dedent(f"""
                [Node] RootNode
                [Children] {[c.path for c in self.children]}
            """)
            
    def display(self):
        print(self)

