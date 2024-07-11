import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import nn


class Model(nn.Module):
    def __init__(self, n_observations, n_actions, batch_size) -> None:
        super().__init__()
        self.layer1 = nn.Linear(n_observations, batch_size)
        self.layer2 = nn.Linear(batch_size, 8)
        self.layer3 = nn.Linear(8, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class NeuralNetwork:
    def __init__(
        self,
        observation_space: int,
        action_space: int,
        batch_size: int,
        model: nn.Sequential | None = None
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        if model is None:
            self.model = Model(observation_space, action_space, batch_size)
        else:
            self.model = model


    @staticmethod
    def plot_durations(show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
