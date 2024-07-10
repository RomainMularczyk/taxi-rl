from pydantic import BaseModel
from lib.models.Action import Action
from lib.models.GameStatus import GameStatus


class EpisodeMetrics(BaseModel):
    """
    Representation of a model evaluation metrics.
    """
    steps: int
    cumulative_reward: float


class StepResult(BaseModel):
    """
    Representation of a step result.
    """
    action: Action
    game_status: GameStatus
    immediate_reward: float


class Result(BaseModel):
    """
    Representation of a bellman step result.
    """
    action: Action
    reward: float
    bellman: float
    game_status: GameStatus


class AgreggatedMetrics(BaseModel):
    """
    Representation of aggregated metrics over many episodes.
    """
    mean: float
    std: float
    max: float
    min: float
