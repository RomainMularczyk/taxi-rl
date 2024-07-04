from __future__ import annotations
from enum import Enum
from typing import List, Tuple
from pydantic import BaseModel
from numpy import array, ndarray
from lib.errors.NotLegalActionException import NotLegalActionException
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

        Raises
        ------
        NotLegalActionException
            If the letter is not contained between 'A' and 'F'.
        """
        if 'A' <= letter <= 'F':
            return Action(ord(letter) - 65)
        else:
            raise NotLegalActionException(
                f"The letter '{letter}' representing the action is not legal."
                " It should be contained between 'A' and 'F'."
            )

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
    def from_human_readable(action: str | None) -> Action:
        """
        Convert the human readable action format to its index.

        Parameters
        ----------
        action: str | None
            The human readable action value.

        Returns
        -------
        int
            The index of the action.

        Raises
        ------
        NotLegalActionException
            If the action is not legal.
        """
        if action == "SOUTH":
            return Action.SOUTH
        elif action == "NORTH":
            return Action.NORTH
        elif action == "WEST":
            return Action.WEST
        elif action == "EAST":
            return Action.EAST
        elif action == "PICK_UP":
            return Action.PICK_UP
        elif action == "DROP_OFF":
            return Action.DROP_OFF
        else:
            raise NotLegalActionException

    @staticmethod
    def legal_actions(info: EnvironmentInfo) -> Tuple[Action]:
        """
        Convert a Gymnasium indexed action into human readable format.

        Parameters
        ----------
        info: EnvInfo
            Gymnasium information at a given step.

        Returns
        -------
        Tuple[Action]
            A tuple of available actions at a given step.
        """
        actions = []
        for index, action in enumerate(info.action_mask):
            if action == 1:
                actions.append(list(Action)[index])
        return tuple(actions)


class ActionProbabilities(BaseModel):
    """
    Action associated with probabilities.
    """
    SOUTH: float | None = None
    NORTH: float | None = None
    EAST: float | None = None
    WEST: float | None = None
    PICK_UP: float | None = None
    DROP_OFF: float | None = None


class ActionWithReward(BaseModel):
    """
    Action associated with the reward we get from the environment
    by taking that action.
    """
    action: Action
    probability: float | None = None
    reward: float

    @staticmethod
    def flatten(list_action_reward: List[ActionWithReward]) -> ndarray:
        """
        Transform a list of actions associated with their reward and
        probability to a two dimension array. The first row contains
        rewards and the second row contains the associated probabilities.

        Returns
        -------
        ndarray
            The flattened reward-probability mapping for a set of actions.
        """
        rewards = []
        probabilities = []
        for el in list_action_reward:
            rewards.append(el.reward)
            probabilities.append(el.probability)
        return array([rewards, probabilities])
