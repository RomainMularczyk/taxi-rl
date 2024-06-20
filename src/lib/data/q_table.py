import numpy as np
from typing import List, Tuple
from pandas import DataFrame
from lib.models.Action import Action


class QTable:
    """
    Representation of a Q-table. Its shape is defined by :
    - The observation space (as lines)
    - The action space (as columns)
    """

    def __init__(self, observation_space: int, action_space: int):
        if type(observation_space) is int:
            self._values = np.zeros((observation_space, action_space))
        else:
            raise ValueError(
                "Q-Table can only be 2-dimensional."
                "You shoud provide two integer values for :\n"
                "- observation_space as lines\n"
                "- action_space as columns"
            )

    def __getitem__(
        self,
        index: Tuple[int, int] | List[int]
    ) -> float:
        if type(index) is tuple or type(index) is list:
            return self._values[index]  # type: ignore
        else:
            raise ValueError(
                "Index values can only be either :\n"
                "- Array of integers\n"
                "- Tuple of integers"
            )

    def __setitem__(
        self,
        index: Tuple[int, int] | List[int],
        value: float
    ):
        if type(value) is float:
            if type(index) is tuple or type(index) is list:
                self._values[*index] = value
            else:
                raise ValueError("Index should be either a list or a tuple.")
        else:
            raise ValueError("Q-Table can only contain float numbers.")

    def update(self, state_index: int, action_index: int) -> None:
        """
        Updates the Q-table.

        Parameters
        ----------
        state_index: int
            The state index.
        action_index: int
            The action index.
        """
        self._values[state_index, action_index]

    def print_table(self) -> DataFrame:
        """
        Return a pandas representation of the Q-table.

        Returns
        -------
        DataFrame
            A DataFrame representing the Q-table.
        """
        actions = [action.name for action in list(Action)]
        return DataFrame(self.values, columns=actions)  # type: ignore
