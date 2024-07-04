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
            state, info = self.env.reset(seed=seed)
        else:
            state, info = self.env.reset()
        self.state = state
        self.info = EnvironmentInfo(**info)

    def render(self) -> None:
        """
        Display the game environment.
        """
        logger.info(self.env.env.render())

    def back_to(self, state: int) -> None:
        """
        Resets the game environement in a given state.

        Parameters
        ----------
        state: int
            The state in which to reset the game environment.
        """
        if state < self.env.observation_space.n and state > 0:  # type: ignore
            self.env.env.unwrapped.s = state  # type: ignore
            self.state = state
        else:
            raise ValueError(
                "States value should be contained between 0 and"
                f" {self.env.observation_space.n}."  # type: ignore
            )

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
