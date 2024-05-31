from enum import Enum
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
    def available_actions(info: EnvironmentInfo):
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
        for index, action in enumerate(info["action_mask"]):
            if action == 1:
                actions.append(list(Action)[index].name)
        return tuple(actions)
