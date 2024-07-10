import numpy as np
from typing import Tuple
from gymnasium.wrappers.time_limit import TimeLimit as GymnasiumGameEnvironment
from lib.models.Action import Action, ActionProbabilities, ActionWithReward
from lib.models.GameStatus import GameStatus
from lib.policies.Policy import Policy


class RandomSamplePolicy(Policy):
    """
    Sample randomly the next action to take from all
    the theoretical actions (including those that are
    not 'legal' moves, i.e. : driving into a wall).

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
    ):
        super().__init__(game_env=game_env, seed=seed)
        self.seed = seed
        self.type = Policy.Type.PROBABILISTIC

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

    @staticmethod
    def actions_probability() -> ActionProbabilities:
        """
        The actions associated with their outcome probability.

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
        return ActionProbabilities(**action_prob)

    def possible_actions(self) -> Tuple[Action, ...]:
        """
        Return all the next available actions following the policy.

        Returns
        -------
        Tuple[Action, ...]
            A list of all possible actions to take from the current state.
        """
        return tuple(list(Action))

    def next_action(  # type: ignore
        self,
        mask: np.ndarray | None = None
    ) -> ActionWithReward:
        """
        Choose the optimal next action.

        Parameters
        ----------
        mask: ndarray | None, default=None
            The possible actions into which to sample.

        Returns
        -------
        ActionWithReward
            The action associated with both :
            - Its immediate reward
            - Its probability of occurring.
            - The current game status (terminated, truncated, running)
        """
        if mask is not None:
            action = self.game_env.env.action_space.sample(mask)
        else:
            action = self.game_env.env.action_space.sample()

        new_state, reward, term, trunc, _ = self.game_env.env.step(action)
        self.game_env.state = new_state

        if term:
            return ActionWithReward(
                action=Action(action),
                reward=float(reward),
                probability=float(1/6),
                game_status=GameStatus.TERMINATED
            )
        elif trunc:
            return ActionWithReward(
                action=Action(action),
                reward=float(reward),
                probability=float(1/6),
                game_status=GameStatus.TRUNCATED
            )
        else:
            return ActionWithReward(
                action=Action(action),
                reward=float(reward),
                probability=float(1/6),
                game_status=GameStatus.RUNNING
            )
