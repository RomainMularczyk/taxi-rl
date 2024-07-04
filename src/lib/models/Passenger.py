from __future__ import annotations
from typing import Tuple
from enum import Enum


class Passenger(Enum):
    """
    The set of passenger locations available in the game environment.
    """
    RED = 0
    GREEN = 1
    YELLOW = 2
    BLUE = 3
    TAXI = 4

    @staticmethod
    def location(
        state: int,
        absolute: bool = False
    ) -> Passenger | Tuple[int, int] | None:
        """
        Return the location of the passenger given the current
        state of the game environment.

        Parameters
        ----------
        state: int
            The state of the game environment.
        absolute: bool, default=False
            Return the absolute coordinate of the passenger location.

        Returns
        -------
        Passenger | Tuple[int, int] | None
            The passenger location. If in absolute mode, returns
            the coordinates in which the passenger is located. It will
            return 'None' if the passenger is in the taxi.
        """
        passenger_location = Passenger((state // 4) % 5)
        if absolute:
            if passenger_location == Passenger.BLUE:
                return (4, 3)
            elif passenger_location == Passenger.GREEN:
                return (0, 4)
            elif passenger_location == Passenger.RED:
                return (0, 0)
            elif passenger_location == Passenger.YELLOW:
                return (4, 0)
            else:
                return None
        return passenger_location

    @staticmethod
    def to_human_readable(passenger: int) -> Passenger:
        """
        Convert the passenger location to a human readable format.

        Parameters
        ----------
        passenger: int
            The index of the destination.

        Returns
        -------
        Passenger
            The passenger location in a human readable format.
        """
        return Passenger(passenger)

    @staticmethod
    def from_human_readable(passenger: str) -> int:
        """
        Convert the human readable passenger location format to its index.

        Parameters
        ----------
        passenger: Destination
            The human readable destination value.

        Returns
        -------
        int
            The index of the passenger location.
        """
        return Passenger(passenger).value
