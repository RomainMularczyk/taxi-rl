import numpy as np
from torch import nn


class NeuralNetwork:
    """"""

    def __init__(
        self,
        observation_space: int,
        action_space: int,
        model: nn.Sequential | None = None
    ) -> None:
        super().__init__()
        self.input_size = observation_space * action_space
        if model is None:
            self.model = nn.Sequential(
                nn.Linear(observation_space * action_space, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 6)
            )
        else:
            self.model = model

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the neural network forward pass.

        Parameters
        ----------
        x: np.ndarray
            The input data to the neural network.

        Returns
        -------
        np.ndarray
            The output vector of the neural network.
        """
        if len(x) != self.input_size:
            raise ValueError(f"The input size should be {self.input_size}.")
        return nn.Softmax(dim=0)(self.model(x))

    def backward():
        pass
