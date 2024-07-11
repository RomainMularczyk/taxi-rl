from enum import Enum
from typing import Literal, Union


class GameStatus(Enum):
    """
    Represents the status of the game environment.
    """
    TERMINATED = "term"
    TRUNCATED = "trunc"
    EXITED = "exit"
    RUNNING = "run"


GameExitStatus = Union[
    Literal[GameStatus.TERMINATED], Literal[GameStatus.TRUNCATED]
]
