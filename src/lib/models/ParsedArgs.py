from pydantic import BaseModel


class ParsedArgs(BaseModel):
    training: bool
    episodes: int
    gamma: float
    #policy: Policy
