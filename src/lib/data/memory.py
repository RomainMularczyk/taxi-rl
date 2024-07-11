import torch
import numpy as np
from typing import List, Tuple
from random import sample
from collections import deque, namedtuple
from pydantic import BaseModel
from lib.models.Action import Action


class Transition(BaseModel):
    """
    Representation of a transition between two states.
    """
    state: int
    action: Action
    next_state: int | None
    reward: float


class Memory:
    """
    Representation of the memory of all states sampled in the
    game environment following a given policy.
    """

    def __init__(self, capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)

    def __repr__(self) -> str:
        return str(list(self.memory))

    def push(self, transition: Transition):
        """
        Save a transition in the memory.

        Parameters
        ----------
        transition: Transition
        """
        self.memory.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        """
        Pick a random batch in the memory.

        Parameters
        ----------
        batch_size: int
            Size of the batch sampled from the memory.

        Returns
        -------
        List[Transition]
            Sampled batch.
        """
        return sample(self.memory, batch_size)

    def __len__(self) -> int:
        """
        Return the number of elements in memory.
        """
        return len(self.memory)

    @staticmethod
    def one_hot(input: int, out_shape: int) -> torch.Tensor:
        """
        One-hot encode a given state or action.

        Parameters
        ----------
        input: int
            A given input.
        out_shape: int
            The output shape of the one-hot encoded vector.

        Returns
        -------
        torch.Tensor
            A one-hot encoded vector.
        """
        one_hot = torch.zeros(out_shape)
        one_hot[input] = 1.0
        return one_hot

    @staticmethod
    def prepare_batch(
        batch: List[Transition]
    ) -> Tuple[List[torch.Tensor | None], ...]:
        Batch = namedtuple(
            "batch",
            ("state", "action", "reward", "next_state")
        )
        action_batch = []
        reward_batch = []
        state_batch = []
        next_state_batch = []
        for sample in batch:
            action_batch.append(Memory.one_hot(sample.action.value, 6))
            state_batch.append(Memory.one_hot(sample.state, 500))
            reward_batch.append(sample.reward)
            if sample.next_state is None:
                next_state_batch.append(None)
            else:
                next_state_batch.append(Memory.one_hot(sample.next_state, 500))
        return Batch(state_batch, action_batch, reward_batch, next_state_batch)

    @staticmethod
    def action_to_index(actions: List[torch.Tensor]) -> torch.Tensor:
        idx = []
        for action in actions:
            idx.append(torch.argmax(action))
        return torch.Tensor(idx).long()
