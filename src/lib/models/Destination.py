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
