from torch import Tensor
from torch.nn import MSELoss
from lib.data.neural_network import NeuralNetwork


class DeepQLearning:

    def __init__(
        self,
        observation_space: int,
        action_space: int,
        data: NeuralNetwork | None = None,
    ):
        if data is None:
            self.data = NeuralNetwork(
                observation_space, action_space
            )
        else:
            self.data = data

    def train():
        pass

    def loss(
        self,
        y: Tensor,
        y_pred: Tensor
    ):
        return MSELoss()(y_pred, y)
