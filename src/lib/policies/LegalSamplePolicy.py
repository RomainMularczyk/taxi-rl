import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit as GymnasiumGameEnvironment
from lib.models.Action import Action, ActionProbabilities
from lib.policies.Policy import Policy


class LegalSamplePolicy(Policy):
    """
    Sample randomly the next action to take from all
    the available actions in a given state 's' (excluding
    those that are not 'legal' moves, i.e. : driving into
    a wall).

    Attributes
    ----------
    p: float
        The probability of a given action to be taken.
    """

    def __init__(
        self,
        env: GymnasiumGameEnvironment,
        seed: int | None = None
    ) -> None:
        super().__init__(env=env, seed=seed)
        self.reward = 0.0

    def actions_probability(self) -> ActionProbabilities:
        action_names = [
            action.name for action, legal in zip(
                list(Action), self.env.info.action_mask
            ) if legal == 1
        ]
        num_legal_actions = len(
            self.env.info.action_mask[self.env.info.action_mask == 1]
        )
        action_prob = {
            action: prob for action, prob in zip(
                action_names, list(
                    np.full(num_legal_actions, 1/num_legal_actions)
                )
            )
        }
        return action_prob  # type: ignore

    def next_action(
        self,
        human_readable: bool = False
    ):
        """
        Choose the optimal next action.

        Parameters
        ----------
        human_readable: bool, default=False
            Output the action chosen in a human readable format.

        Returns
        -------
        int | str
            The action chosen.
        """
        action = self.env.env.action_space.sample()
        if human_readable:
            return Action.to_human_readable(action)
        return action
