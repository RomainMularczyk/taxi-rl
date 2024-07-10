import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit as GymnasiumGameEnvironment
from lib.models.Action import Action, ActionWithReward
from lib.models.GameStatus import GameStatus
from lib.policies.Policy import Policy


class GreedyPolicy(Policy):
    """
    Pick the next action to take as the one providing the highest
    immediate reward.

    Attributes
    ----------
    type: Policy.Type
        The type of policy. It can be either :
        - Probabilistic
        - Deterministic
    """

    def __init__(
        self,
        game_env: GymnasiumGameEnvironment,
        seed: int | None = None
    ) -> None:
        super().__init__(game_env=game_env, seed=seed)
        self.seed = seed

    def reset_hyperparameters(self, reset_env: bool = False) -> None:
        """
        Resets the environment and the hyperparameters of the policy.

        Parameters
        ----------
        reset_env: bool, default=False
            Decide if the environment should be also reset.
        """
        if reset_env:
            self.game_env.env.reset(seed=self.seed)

    def next_action(self, action: Action) -> ActionWithReward:  # type: ignore
        """
        Returns the next action maximizing the immediate reward.

        Parameters
        ----------
        action: Action
            The next action according to the best Q-value.
        mask: np.ndarray | None, default=None
            The possible actions into which to sample.

        Returns
        -------
        ActionWithReward
            The action associated with both :
            - Its immediate reward
            - Its probability of occurring.
            - The current game status (terminated, truncated, running)

        Raises
        ------
        ValueError
            If no optimal action could be found.
        """
        return self.take_action(action)
