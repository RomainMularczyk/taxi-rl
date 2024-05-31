from numpy import ndarray
from typing import TypedDict


class EnvironmentInfo(TypedDict):
    """
    Represent the game environment informations at each step.
    """
    action_mask: ndarray
    prob: float
