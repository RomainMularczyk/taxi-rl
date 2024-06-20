from pydantic import BaseModel
from lib.models.Policies import Policies


class ParsedArgs(BaseModel):
    training: bool
    episodes: int
    gamma: float
    policy: Policies

    class Config:
        arbitrary_types_allowed = True
