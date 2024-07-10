from __future__ import annotations
from enum import Enum
from lib.errors.NotLegalStageException import NotLegalStageException


class Stage(Enum):
    """
    Split of the different sets of stages during the game.
    """
    PICK = 0
    DROP = 1

    @staticmethod
    def to_human_readable(stage: int) -> str:
        """
        Convert the stage to human readable format.

        Parameters
        ----------
        stage: int
            The index of the stage.

        Returns
        -------
        str
            The stage in a human readable format.
        """
        return Stage(stage).name
    
    @staticmethod
    def from_human_readable(stage: str | None) -> Stage:
        """
        Convert the human readable stage format to its index.

        Parameters
        ----------
        stage: str | None
            The human readable stage value.

        Returns
        -------
        Stage
            The Stage.

        Raises
        ------
        NotLegalActionException
            If the stage is not legal.
        """
        if stage == "PICK":
            return Stage.PICK
        elif stage == "DROP":
            return Stage.DROP
        else:
            raise NotLegalStageException