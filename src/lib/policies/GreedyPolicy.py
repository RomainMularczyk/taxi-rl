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
        self.type = Policy.Type.DETERMINISTIC

    def reset_hyperpameters(self) -> None:
        """
        Resets the environment and the hyperparameters of the policy.
        """
        self.game_env.env.reset()

    def next_action(  # type: ignore
        self,
        action: Action,
        mask: np.ndarray | None = None,
    ) -> ActionWithReward:
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
        if mask is not None:
            action = self.game_env.env.action_space.sample(mask)
            new_state, reward, term, trunc, _ = self.game_env.env.step(action)
            self.game_env.state = new_state
            if term:
                return ActionWithReward(
                    action=Action(action),
                    reward=float(reward),
                    probability=1.0,
                    game_status=GameStatus.TERMINATED
                )
            elif trunc:
                return ActionWithReward(
                    action=Action(action),
                    reward=float(reward),
                    probability=1.0,
                    game_status=GameStatus.TRUNCATED
                )
            else:
                return ActionWithReward(
                    action=Action(action),
                    reward=float(reward),
                    probability=1.0,
                    game_status=GameStatus.RUNNING
                )
        else:
            return self.take_action(action)
