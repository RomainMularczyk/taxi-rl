import logging
from pydantic import TypeAdapter, ValidationError
from lib.environment.environment import GameEnvironment
from lib.models.Action import Action
from lib.models.EnvironmentInfo import EnvironmentInfo


class RandomSamplePolicy:
    """
    Sample randomly the next action to take from all
    the theoretical actions (including those that are
    not 'legal' moves, i.e. : driving into a wall).

    Attributes
    ----------
    p: float
        The probability of a given action to be taken.
    env: GameEnvironment
        The game environment.
    reward: float
        The reward collected for taking a given step.
    state: int
        The seed representing a given state of the game environment.
    """

    def __init__(
        self,
        env: GameEnvironment,
        p: float = 1/6
    ):
        self.p = p
        self.env = env
        self.reward = 0.0

    def next_action(
        self,
        human_readable: bool = False,
        render: bool = True,
        display_actions: bool = False
    ):
        """
        Choose the optimal next action.

        Parameters
        ----------
        env: GameEnvironement
            The game environement.
        human_readable: bool, default=False
            Output the action chosen in a human readable format.
        render: bool, default=True
            Defines if the step should be displayed.
        display_actions: bool, default=False
            Defines if the available actions at a given step should be
            displayed.

        Returns
        -------
        int | str
            The action chosen.
        """
        action = self.env.env.action_space.sample()

        observation, reward, terminated, truncated, info = self.env.env.step(
            action
        )
        self.state = observation[0]

        if render:
            logging.info(self.env.env.render())
        if display_actions:
            try:
                TypeAdapter(EnvironmentInfo).validate_python(info)
                logging.info(Action.available_actions(info))  # type: ignore
            except ValidationError:
                logging.error(
                    "The environment information does"
                    " not have the expected format."
                )

        self.reward += float(reward)

        if human_readable:
            return Action.to_human_readable(action)
        return action
