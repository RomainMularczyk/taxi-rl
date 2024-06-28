from __future__ import annotations
from enum import Enum
from typing import TypedDict
from lib.models.EnvironmentInfo import EnvironmentInfo


class Action(Enum):
    """
    The set of actions available in the game environment.
    """
    SOUTH = 0
    NORTH = 1
    EAST = 2
    WEST = 3
    PICK_UP = 4
    DROP_OFF = 5

    def to_letter(self) -> str:
        """
        Convert the action to a corresponding letter.

        Returns
        -------
        str
            The letter corresponding to the action.
        """
        return chr(65 + self.value)

    @staticmethod
    def from_letter(letter: str) -> Action:
        """
        Convert a letter to the corresponding action.

        Parameters
        ----------
        letter: str
            The letter representing the action.

        Returns
        -------
        Action
            The action corresponding to the letter.
        """
        if 'A' <= letter <= 'F':
            return Action(ord(letter) - 65)

    @staticmethod
    def to_human_readable(action: int) -> str:
        """
        Convert the action to human readable format.

        Parameters
        ----------
        action: int
            The index of the action.

        Returns
        -------
        str
            The action in a human readable format.
        """
        return Action(action).name

    @staticmethod
    def from_human_readable(action: str) -> int:
        """
        Convert the human readable action format to its index.

        Parameters
        ----------
        action: str
            The human readable action value.

        Returns
        -------
        int
            The index of the action.
        """
        return Action(action).value

    @staticmethod
    def legal_actions(info: EnvironmentInfo):
        """
        Convert a Gymnasium indexed action into human readable format.

        Parameters
        ----------
        info: EnvInfo
            Gymnasium information at a given step.

        Returns
        -------
        Tuple[str]
            A tuple of available actions at a given step.
        """
        actions = []
        for index, action in enumerate(info.action_mask):
            if action == 1:
                actions.append(list(Action)[index].name)
        return tuple(actions)


class ActionProbabilities(TypedDict):
    """
    Action associated with probabilities.
    """
    SOUTH: float
    NORTH: float
    EAST: float
    WEST: float
    PICK_UP: float
    DROP_OFF: float
