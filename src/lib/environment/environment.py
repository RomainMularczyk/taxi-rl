import logging
from typing import Tuple
from numpy import ndarray
from pydantic import TypeAdapter, ValidationError
from gymnasium.wrappers.time_limit import TimeLimit as GymnasiumGameEnvironment
from lib.models.EnvironmentInfo import EnvironmentInfo
from lib.models.Action import Action
from lib.models.Policies import Policies


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
        reward: float,
        policy: Policies,
        q_table: ndarray,
        seed: int | None = None,
    ):
        self.env = env
        self.reward = reward
        self.policy = policy

        if seed:
            self.env.reset(seed=seed)
        else:
            self.env.reset()

        if q_table:
            self.q_table = q_table

    def do_step(
        self,
        previous_rewards: float,
        render: bool = False,
        available_actions: bool = False,
    ) -> None:
        """
        Compute a game environment step.

        Parameters
        ----------
        env: GameEnv
            The game environment.
        previous_rewards: int
            The accumulation of all the rewards following a given trajectory.
        render: bool, default=False
            Defines if the step should be graphically rendered.
        available_actions: bool, default=False
            Defines if the available actions at a given step should be
            displayed.
        """
        action = self.env.action_space.sample()
        observation, reward, terminated, truncated, info = self.env.step(
            action
        )
        self.state = observation[0]
        print(self.state)

        if render:
            print(self.env.render())
        elif available_actions:
            try:
                TypeAdapter(EnvironmentInfo).validate_python(info)
                print(Action.available_actions(info))  # type: ignore
            except ValidationError:
                logging.error(
                    "The environment information does"
                    "not have the expected format."
                )
        else:
            self.reward += float(reward)

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
