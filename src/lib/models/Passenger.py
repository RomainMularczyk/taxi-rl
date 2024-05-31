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
    def to_human_readable(passenger: int) -> str:
        """
        Convert the passenger location to a human readable format.

        Parameters
        ----------
        passenger: int
            The index of the destination.

        Returns
        -------
        str
            The passenger location in a human readable format.
        """
        return Passenger(passenger).name

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
