from __future__ import annotations
from typing import Tuple
from enum import Enum


class Destination(Enum):
    """
    The set of destinations available in the game environment.
    """
    RED = 0
    GREEN = 1
    YELLOW = 2
    BLUE = 3

    @staticmethod
    def location(
        state: int,
        absolute: bool = False
    ) -> Destination | Tuple[int, int] | None:
        destination_location = Destination(state % 4)
        if absolute:
            if destination_location == Destination.BLUE:
                return (4, 3)
            elif destination_location == Destination.GREEN:
                return (0, 4)
            elif destination_location == Destination.RED:
                return (0, 0)
            elif destination_location == Destination.YELLOW:
                return (4, 0)
            else:
                return None
        return destination_location

    @staticmethod
    def to_human_readable(destination: int) -> str:
        """
        Convert the destination to a human readable format.

        Parameters
        ----------
        destination: int
            The index of the destination.

        Returns
        -------
        str
            The destination in a human readable format.
        """
        return Destination(destination).name

    @staticmethod
    def from_human_readable(destination: str) -> int:
        """
        Convert the human readable destination format to its index.

        Parameters
        ----------
        destination: Destination
            The human readable destination value.

        Returns
        -------
        int
            The index of the destination.
        """
        return Destination(destination).value
