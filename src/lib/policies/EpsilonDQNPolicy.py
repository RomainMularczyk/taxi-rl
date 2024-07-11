import numpy as np
from gymnasium.wrappers.time_limit import TimeLimit as GymnasiumGameEnvironment
from lib.data.neural_network import Model
from lib.models.Action import Action, ActionWithReward
from lib.models.GameStatus import GameStatus
from lib.policies.DQNPolicy import DQNPolicy


class EpsilonDQNPolicy(DQNPolicy):

    def __init__(
        self,
        game_env: GymnasiumGameEnvironment,
        policy_network: Model,
        epsilon: float = 1.0,
        decay_rate: float = 0.01,
        legal: bool = False,
        seed: int | None = None
    ) -> None:
        super().__init__(
            game_env=game_env,
            seed=seed,
            policy_network=policy_network
        )  # type: ignore
        self.epsilon = epsilon
        self.init_epsilon = epsilon
        self.decay_rate = decay_rate
        self.legal = legal
        self.seed = seed
        self.policy_network = policy_network

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

    def _exploit(self) -> ActionWithReward:
        return super().next_action()

    def next_action(self) -> ActionWithReward:
        self._steps += 1
        random = np.random.uniform(0.0, 1.0)

        if random >= self.epsilon:
            self._update_epsilon()
            return self._exploit()
        else:
            self._update_epsilon()
            return self._explore()
