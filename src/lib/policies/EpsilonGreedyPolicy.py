import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit as GymnasiumGameEnvironment
from lib.models.Action import Action, ActionWithReward
from lib.models.GameStatus import GameStatus
from lib.policies.GreedyPolicy import GreedyPolicy
from lib.policies.Policy import Policy


class EpsilonGreedyPolicy(GreedyPolicy):
    """
    Pick the next action to take either by exploring (i.e. picking
    the next action randomly) or exploiting (i.e. picking the next
    action by maximizing the immediate reward).

    Attributes
    ----------
    type: Policy.Type
        The type of policy. It can be either :
        - Probabilistic
        - Deterministic
    legal: bool, default=True
        Decide whether to choose between legal actions or all possible actions
        when exploring instead of exploiting.
    epsilon: float, default=0
        The epsilon factor deciding between exploitation and exploration.
    decay_rate: float, default=0.1
        The rate at which the epsilon value decreases over steps.
    steps: int
        The number of steps taken with the policy.
    """

    def __init__(
        self,
        game_env: GymnasiumGameEnvironment,
        legal: bool = False,
        epsilon: float = 1.0,
        decay_rate: float = 0.001,
        seed: int | None = None
    ):
        super().__init__(game_env=game_env, seed=seed)
        self.seed = seed
        self.legal = legal
        self.init_epsilon = epsilon
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self._steps = 0
        self.type = Policy.Type.DETERMINISTIC

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
        self._steps = 0
        self.epsilon = self.init_epsilon

    def _update_epsilon(self) -> None:
        """
        Updates the epsilon value following the decay rate.
        """
        self.epsilon = self.epsilon * np.exp(-self.decay_rate * self._steps)

    def _explore(self) -> ActionWithReward:
        """
        Choose the next action by picking randomly among either :
        - The next legal actions (legal=True)
        - All the actions defined in the entire action space (legal=False)

        Returns
        -------
        ActionWithReward
            The action chosen associated with the reward from taking it.
        """
        if self.legal:
            legal_actions = Action.legal_actions(self.game_env.info)
            action = np.random.choice(np.array(legal_actions))
            action = self.take_action(action)
            if action == GameStatus.TERMINATED:
                return ActionWithReward(
                    action=action.action,
                    reward=action.reward,
                    probability=float(1/len(legal_actions)),
                    game_status=GameStatus.TERMINATED
                )
            elif action == GameStatus.TRUNCATED:
                return ActionWithReward(
                    action=action.action,
                    reward=action.reward,
                    probability=float(1/len(legal_actions)),
                    game_status=GameStatus.TRUNCATED
                )
            else:
                return ActionWithReward(
                    action=action.action,
                    reward=action.reward,
                    probability=float(1/len(legal_actions)),
                    game_status=GameStatus.RUNNING
                )
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

    def _exploit(self, action: Action) -> ActionWithReward:
        """
        Choose the next action by following the greedy policy behavior.

        Returns
        -------
        ActionWithReward
            The action chosen associated with the reward from taking it.

        """
        return super().next_action(action)

    def next_action(
        self,
        action: Action,
        mask: np.ndarray | None = None,
    ) -> ActionWithReward:
        """
        Returns the next action following the epsilon greedy tradeoff.

        Returns
        -------
        ActionWithReward | GameExitStatus
            The action associated with its immediate reward and its
            probability of occurring. If the game is ended prematurely,
            return a GameStatus.
        """
        self._steps += 1
        if mask is not None:
            action = self.game_env.env.action_space.sample(mask)
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
                    probability=1.0,
                    game_status=GameStatus.RUNNING
                )
        random = np.random.uniform(0.0, 1.0)

        if random >= self.epsilon:
            self._update_epsilon()
            return self._exploit(action)
        else:
            self._update_epsilon()
            return self._explore()
