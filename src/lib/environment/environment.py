from typing import Any, List, Tuple
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
        self.initial_state = state
        self.state = state
        self.initial_info = EnvironmentInfo(**info)
        self.info = EnvironmentInfo(**info)

    def render(self) -> List[Any] | None:
        """
        Display the game environment.
        """
        return self.env.render()

    def back_to(self, state: int) -> None:
        """
        Resets the game environement in a given state.

        Parameters
        ----------
        state: int
            The state in which to reset the game environment.
        """
        if state < self.env.observation_space.n and state >= 0:  # type: ignore
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

    def passenger_pickedup(self, state: int) -> bool:
        """
        Know if the passenger is in the taxi.

        Returns
        -------
        bool
            True if the passenger is in the taxi.
        """
        # _, _, passenger_location, _ = self.env.env.decode(state)
        _, _, passenger_location, _ = self.env.unwrapped.decode(state)  # type: ignore
        passenger_in_taxi = (passenger_location == 4)
        return passenger_in_taxi

    def passenger_droppedoff(self, state: int | None) -> bool:
        """
        Know if the passenger has been dropped off the taxi at the
        winning state.

        Returns
        -------
        bool
            True if the passenger has been dropped in the desired place.
            This means you had a reward of +20 for this action.
        """
        if state is None:
            raise ValueError(
                "You can't know if the passenger dropped off if"
                " the state is unknown."
            )
        _, _, passenger_location, destination = self.env.unwrapped.decode(  # type: ignore
            state
        )
        return passenger_location == destination
