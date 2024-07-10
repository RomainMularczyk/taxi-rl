from __future__ import annotations
import textwrap
from typing import List, Tuple
from lib.models.Action import Action
from lib.models.EnvironmentInfo import EnvironmentInfo


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
        depth: int,
        action: Action | None = None, 
        parent: Node | None = None, 
        state: int | None = None, 
        env_info: EnvironmentInfo | None = None, 
        children: List[Node] | List = [],
        path: str | None = '',
        reward: float | None = .0,
        cumul_reward: float | None = .0
    ):
        self.action = action
        self.path = path
        self.depth = depth
        self.state = state
        self.parent = parent
        self.children = children
        self.env_info = env_info
        self.reward = reward
        self.cumul_reward = cumul_reward

    
    def update_reward(self, reward: float) -> None:
        """
        Update the reward and cumulative reward from the Node.

        Attributes
        ----------
        raward: int
            The reward obtained after taking the `self.action` on the game env.
        """
        self.reward = reward
        self.cumul_reward = reward + self.parent.cumul_reward

    def update_children_with(self, env_info: EnvironmentInfo) -> None:
        """
        Delete the Node's children that are illegal action according to the action_mask.

        Attributes
        ----------
        action_mask: list[int]
            The action mask of the state associated to this Node and its action.
        """
        allowed_actions: Tuple[Action] = Action.legal_actions(env_info)
        allowed_children = [child for child in self.children if child.action in allowed_actions]
        self.children = allowed_children


    def __repr__(self):
        try:
            return textwrap.dedent(f"""
                [Node] {self.fullpath}
                [Parent] {self.parent.fullpath}
                [Children] {[c.path for c in self.children]}
                [State] {self.state}
                [Reward/Cumul] {self.reward}/{self.cumul_reward}
            """)
        except AttributeError:
            return textwrap.dedent(f"""
                [Node] RootNode
                [Children] {[c.path for c in self.children]}
            """)
            
    def display(self):
        print(self)

    @property
    def fullpath(self) -> str:
        """
        Generate the fullpath of this Node recursivly.

        Returns
        -------
        fullpath
            The full path of the Node, meaning the path of 
            each its ancesters that leads to him.
        """
        if self.parent is None:
            return self.path
        return f"{self.parent.fullpath}{self.path}"
    
    @property
    def actions(self) -> list[Action]:
        """
        Generate a list of Action to take in order to be in this node state.
        
        Returns
        -------
        list[Action]
            A list of Action to take in order to be in this node state.
        """
        actions = []
        for char in self.fullpath:
            if char.isalpha():
                actions.append(Action.from_letter(char))
        return actions

    @property
    def first_layer_parent(self) -> Node:
        """
        Find the oldest parent of a Node that is not the root Node.

        Returns
        -------
        node: Node
            The parent Node of self that is at layer 1.
        """
        if self.depth <= 1 :
            return self
        else: 
            return self.parent.first_layer_parent
