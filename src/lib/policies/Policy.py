from abc import ABCMeta, abstractmethod
from enum import Enum
from gymnasium.wrappers.time_limit import TimeLimit as GymnasiumGameEnvironment
from lib.environment.environment import GameEnvironment
from lib.models.Action import Action, ActionWithReward
from lib.models.GameStatus import GameStatus, GameExitStatus


class Policy(metaclass=ABCMeta):
    """
    Abstract representation of a policy.

    Attributes
    ----------
    game_env: GymnasiumGameEnvironment
        The game environment.
    """

    class Type(Enum):
        PROBABILISTIC = "P"
        DETERMINISTIC = "D"

    def __init__(
        self,
        game_env: GymnasiumGameEnvironment,
        seed: int | None
    ):
        self.game_env = GameEnvironment(env=game_env, seed=seed)

    @abstractmethod
    def next_action(self) -> ActionWithReward:
        """
        Compute the next optimal action.

        Returns
        -------
        ActionWithReward | GameExitStatus
            The action associated with its immediate reward and its
            probability of occurring. If the game is ended prematurely,
            return a GameStatus.
        """
        pass

    def is_game_over(self, action: ActionWithReward | GameExitStatus) -> bool:
        """
        Verify if the game is over.

        Parameters
        ----------
        action: ActionWithReward | GameExitStatus
            The current action taken.

        Returns
        -------
        bool
            Returns True if the game is over.
        """
        if action == GameStatus.TERMINATED:
            return True
        elif action == GameStatus.TRUNCATED:
            return True
        else:
            return False

    def take_action(
        self,
        action: Action | int,
    ) -> ActionWithReward:
        """
        Take a given action (deterministic, probability=1.0).

        Parameters
        ----------
        action: Action | int
            The action taken.

        Returns
        -------
        ActionWithReward
            The action associated with both :
            - Its immediate reward
            - Its probability of occurring.
            - The current game status (terminated, truncated, running)

        Raises
        ------
        TypeError
            If the action is neither an integer or an Action.
        """
        if type(action) is int:
            new_state, reward, term, trunc, _ = self.game_env.env.step(
                action
            )
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
        elif type(action) is Action:
            new_state, reward, term, trunc, _ = self.game_env.env.step(
                action.value
            )
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
            return ActionWithReward(
                action=action,
                reward=float(reward),
                probability=1.0,
                game_status=GameStatus.RUNNING
            )
        else:
            raise TypeError(
                "The action should either be an integer or an Action."
            )
