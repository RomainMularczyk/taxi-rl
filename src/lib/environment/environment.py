from typing import Tuple
from gymnasium.wrappers.time_limit import TimeLimit as GymnasiumGameEnvironment
from lib.models.EnvironmentInfo import EnvironmentInfo
from lib.logs.logger import get_logger

logger = get_logger()


class GameEnvironment:
    """
    Represent the game environement.

    Attributes
    ----------
    env: GameEnv
        The game environment.
    """

    def __init__(
        self,
        env: GymnasiumGameEnvironment,
        seed: int | None = None,
    ):
        self.env = env

        # Init env with or without seed
        if seed:
            state, info = self.env.env.reset(seed=seed)
        else:
            state, info = self.env.env.reset()
        self.state = state
        self.info = EnvironmentInfo(**info)

    def render(self) -> None:
        """
        Display the game environment.
        """
        logger.info(self.env.render())

    def back_to(self, seed: int) -> None:
        """
        Resets the game environement in a previous seed.

        Parameters
        ----------
        seed: int
            The seed to which reset the game environment into.
        """
        self.env.reset(seed=seed)

    def current_taxi_location(self) -> Tuple[int, int]:
        """
        Gets the current taxi location.

        Returns
        -------
        Tuple[int, int]
            The location of the taxi on the map in the following
            format : (row, column).
        """
        row = (self.env.s // 100) % 5
        column = (self.env.s // 20) % 5
        return (row + 1, column + 1)
