import torch
import numpy as np
from tqdm import tqdm
from typing import Callable, List
from lib.data.neural_network import NeuralNetwork
from lib.models.Action import ActionWithReward
from lib.models.GameStatus import GameStatus
from lib.models.Metrics import EpisodeMetrics
from lib.models.Policies import Policies
from lib.data.memory import Memory, Transition
from lib.policies.GreedyPolicy import GreedyPolicy
from lib.policies.EpsilonGreedyPolicy import EpsilonGreedyPolicy


class DeepQLearning:

    def __init__(
        self,
        policy: Policies,
        loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        data: Memory | None = None,
        sample_depth: int = 50,
        gamma: float = 1.0,
        tau: float = 1.0,
        batch_size: int = 128,
    ) -> None:
        if data is None:
            self.data = Memory(128)
        else:
            self.data = data
        self.policy = policy
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.gamma = gamma
        self.sample_depth = sample_depth
        self.tau = tau
        self.batch_size = batch_size
        self.policy_network = NeuralNetwork(
            self.policy.game_env.env.observation_space.n,  # type: ignore
            self.policy.game_env.env.action_space.n,  # type: ignore
            self.batch_size
        )
        self.target_network = NeuralNetwork(
            self.policy.game_env.env.observation_space.n,  # type: ignore
            self.policy.game_env.env.action_space.n,  # type: ignore
            self.batch_size
        )
        self.target_network.model.load_state_dict(
            self.policy_network.model.state_dict()
        )

    def do_episode(self) -> EpisodeMetrics:
        """
        Execute an entire episode. The episode stops either because :
        - The maximum number of steps is reached (200)
        - The game exited because the winning condition was reached
        - The game exited because a losing condition was reached

        Returns
        -------
        EpisodeMetrics
            The number of steps and the cumulative reward.
        """
        self.policy.reset_hyperparameters(reset_env=True)
        steps = 0
        cumulative_reward = 0
        for _ in range(self.sample_depth):
            next = self._generate_one_sample()
            steps += 1
            if next.game_status == GameStatus.TERMINATED \
               or next.game_status == GameStatus.TRUNCATED:
                pass
            cumulative_reward += next.reward
            self._optimize_model()
            self._update_target_network()
        return EpisodeMetrics(
            steps=steps,
            cumulative_reward=cumulative_reward
        )

    def train(self, num_episodes: int) -> List[EpisodeMetrics]:
        metrics = []
        for _ in tqdm(range(num_episodes)):
            metrics.append(self.do_episode())
        return metrics

    def _target_q(
        self,
        reward: torch.Tensor,
        next_state: torch.Tensor
    ) -> torch.Tensor:
        return reward + self.gamma * next_state

    def _update_target_network(self):
        target_network_layers = self.target_network.model.state_dict()
        policy_network_layers = self.policy_network.model.state_dict()
        for layer in policy_network_layers:
            target_network_layers[layer] = policy_network_layers[layer] \
              * self.tau + target_network_layers[layer] * (1 - self.tau)

    def _generate_one_sample(self) -> ActionWithReward:
        """
        Generate one sample from the game environment and
        push it to the memory.

        Returns
        -------
        ActionWithReward
            The action associated with both :
            - Its immediate reward
            - Its probability of occurring.
            - The current game status (terminated, truncated, running)
        """
        current_state = self.policy.game_env.state
        # Verify if the policy needs a Q-Table
        if type(self.policy) is GreedyPolicy \
           or type(self.policy) is EpsilonGreedyPolicy:
            raise TypeError("Policy not compatible with this formulas.")
        else:
            next = self.policy.next_action()  # type: ignore

            if next.game_status == GameStatus.TERMINATED\
               or next.game_status == GameStatus.TRUNCATED:
                self.data.push(
                    Transition(
                        state=current_state,
                        action=next.action,
                        reward=next.reward,
                        next_state=None
                    )
                )
            else:
                self.data.push(
                    Transition(
                        state=current_state,
                        action=next.action,
                        reward=next.reward,
                        next_state=self.policy.game_env.state
                    )
                )
            return next

    def _optimize_model(self) -> None:
        # If not enough samples to train
        if len(self.data) < self.batch_size:
            return

        batch = self.data.sample(self.batch_size)
        states, actions, rewards, n_states = Memory.prepare_batch(batch)
        actions = Memory.action_to_index(actions)
        policy_out = self.policy_network.model(
            torch.Tensor(np.array(states))
        )
        q_values = self.policy_network.model(
            torch.Tensor(np.array(states))
        ).gather(1, actions.unsqueeze(1)).squeeze(0)

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, n_states)),
            dtype=torch.bool
        )
        non_final_next_states = torch.tensor(
            np.array([s for s in n_states if s is not None])
        )

        vs = torch.zeros(self.batch_size)
        with torch.no_grad():
            vs[non_final_mask] = self.target_network.model(
                non_final_next_states
            ).max(1).values

        target_q_values = self._target_q(
            torch.tensor(np.array(rewards, dtype="float32")), vs
        )
        #print(q_values, target_q_values)

        loss = self.loss_function(q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
