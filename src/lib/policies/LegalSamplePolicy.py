import numpy as np
from typing import Tuple
from gymnasium.wrappers.time_limit import TimeLimit as GymnasiumGameEnvironment
from lib.models.Action import Action, ActionProbabilities, ActionWithReward
from lib.models.GameStatus import GameStatus
from lib.policies.Policy import Policy


class LegalSamplePolicy(Policy):
    """
    Sample randomly the next action to take from all
    the available actions in a given state 's' (excluding
    those that are not 'legal' moves, i.e. : driving into
    a wall).

    Attributes
    ----------
    type: Policy.Type
        Defines if the policy is deterministic or probabilistic.
    """

    def __init__(
        self,
        game_env: GymnasiumGameEnvironment,
        seed: int | None = None
    ) -> None:
        super().__init__(game_env=game_env, seed=seed)
        self.seed = seed
        self.type = Policy.Type.PROBABILISTIC

    def reset_hyperparameters(
        self,
        reset_env: bool = False
    ) -> None:
        """
        Resets the environment and the hyperparameters of the policy.

        Parameters
        ----------
        reset_env: bool, default=False
            Decide if the environment should be also reset.
        """
        if reset_env:
            self.game_env.env.reset(seed=self.seed)

    def actions_probability(self) -> ActionProbabilities:
        """
        Returns the probability associated with each action.

        Returns
        -------
        ActionProbabilities
            The probability associated with each legal action.
        """
        legal_actions_names = [
            action.name for action in self.possible_actions()
        ]
        num_legal_actions = len(
            self.game_env.info.action_mask[self.game_env.info.action_mask == 1]
        )
        action_prob = {
            action: prob for action, prob in zip(
                legal_actions_names, list(
                    np.full(num_legal_actions, 1/num_legal_actions)
                )
            )
        }
        return ActionProbabilities(**action_prob)

    def possible_actions(self) -> Tuple[Action, ...]:
        """
        Return all the next available actions following the policy.

        Returns
        -------
        Tuple[Action, ...] | None
            A list of all possible actions to take from the current state.
            Returns None if no action is possible.
        """
        return Action.legal_actions(self.game_env.info)

    def next_action(  # type: ignore
        self,
        mask: np.ndarray | None = None
    ) -> ActionWithReward:
        """
        Choose the optimal next action.

        Parameters
        ----------
        mask: np.ndarray | None, default=None
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
            new_state, reward, term, trunc, _ = self.game_env.env.step(action)
            self.game_env.state = new_state
            if term:
                return ActionWithReward(
                    action=Action(action),
                    reward=float(reward),
                    game_status=GameStatus.TERMINATED
                )
            elif trunc:
                return ActionWithReward(
                    action=Action(action),
                    reward=float(reward),
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
            action = np.random.choice(np.array(self.possible_actions()))
            action = self.take_action(action)
            if action == GameStatus.TERMINATED:
                return ActionWithReward(
                    action=action.action,
                    reward=float(action.reward),
                    game_status=GameStatus.TERMINATED
                )
            elif action == GameStatus.TRUNCATED:
                return ActionWithReward(
                    action=Action(action.action),
                    reward=float(action.reward),
                    game_status=GameStatus.TRUNCATED
                )
            else:
                return ActionWithReward(
                    action=action.action,
                    reward=action.reward,
                    probability=float(
                        1/len(self.possible_actions())
                    ),
                    game_status=GameStatus.RUNNING
                )
