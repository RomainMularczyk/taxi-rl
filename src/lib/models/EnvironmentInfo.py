from numpy import ndarray
from pydantic import BaseModel


class EnvironmentInfo(BaseModel):
    """
    Represent the game environment informations at each step.
    """
    action_mask: ndarray
    prob: float

    class Config:
        arbitrary_types_allowed = True
