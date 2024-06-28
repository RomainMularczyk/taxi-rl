import logging
import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit as GymnasiumGameEnvironment
from pydantic import TypeAdapter, ValidationError
from lib.environment.environment import GameEnvironment
from lib.models.Action import Action, ActionProbabilities
from lib.models.EnvironmentInfo import EnvironmentInfo
from lib.policies.Policy import Policy


class RandomSamplePolicy(Policy):
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
        env: GymnasiumGameEnvironment,
        seed: int | None = None
    ):
        super().__init__(env=env, seed=seed)
        self.reward = 0.0

    def actions_probability(self) -> ActionProbabilities:
        """
        Returns
        -------
        ActionProbabilities
            Action associated with their probabilities.
        """
        action_names = [action.name for action in list(Action)]
        action_prob = {
            action: prob for action, prob in zip(
                action_names, list(np.full(6, 1/6))
            )
        }
        return action_prob  # type: ignore

    def next_action(
        self,
        render: bool = True,
        display_actions: bool = False
    ) -> Action:
        """
        Choose the optimal next action.

        Parameters
        ----------
        env: GameEnvironement
            The game environement.
        render: bool, default=True
            Defines if the step should be displayed.
        display_actions: bool, default=False
            Defines if the available actions at a given step should be
            displayed.

        Returns
        -------
        Action
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

        return Action(action)
