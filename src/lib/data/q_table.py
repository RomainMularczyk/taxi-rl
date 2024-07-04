import numpy as np
import pandas as pd
from typing import List, Tuple
from pandas import DataFrame
from IPython.display import display, HTML
from lib.models.Action import Action


class QTable:
    """
    Representation of a Q-table. Its shape is defined by :
    - The observation space (as lines)
    - The action space (as columns)
    """

    def __init__(
        self,
        observation_space: int,
        action_space: int,
        data: np.ndarray | None = None
    ) -> None:
        err = ValueError(
            "Q-Table can only be 2-dimensional."
            " You shoud provide two integer values for :\n"
            "- 'observation_space' as lines\n"
            "- 'action_space' as columns"
        )
        if data is None:
            try:
                self._values = np.zeros(
                    (int(observation_space), int(action_space))
                )
            except TypeError:
                raise err
        else:
            if data.shape == (observation_space, action_space):
                self._values = data
            else:
                raise err

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
    ) -> None:
        if type(value) is float:
            if type(index) is tuple or type(index) is list:
                self._values[*index] = value
            else:
                raise ValueError("Index should be either a list or a tuple.")
        else:
            raise ValueError("Q-Table can only contain float numbers.")

    def display_table(self, scrollable: bool = True) -> DataFrame | None:
        """
        Return a pandas representation of the Q-table.

        Parameters
        ----------
        scrollable: bool, default=True
            Display the table as scrollable.

        Returns
        -------
        DataFrame | None
            A DataFrame representing the Q-table.
        """
        actions = [action.name for action in list(Action)]
        df = DataFrame(self._values, columns=actions)  # type: ignore
        if scrollable:
            pd.set_option("display.max_rows", None)

            display(
                HTML(
                    f"""<div
                    style='height: 200px'
                    overflow: auto;
                    width: fit-content'
                    >{df.style.to_html()}</div>"""
                )
            )
        else:
            return df
