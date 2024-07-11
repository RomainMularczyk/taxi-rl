import numpy as np
import matplotlib.pyplot as plt
from typing import List
from lib.models.Metrics import AgreggatedMetrics, EpisodeMetrics


def graph_training_evolution(metrics: List[EpisodeMetrics]):
    x = []
    y = []
    for n, episode in enumerate(metrics):
        x.append(n)
        y.append(episode.cumulative_reward)
    plt.plot(x, y)
    plt.show()


def aggregate_metrics(metrics: List[EpisodeMetrics]):
    agg_metrics = [metric.cumulative_reward for metric in metrics]
    return AgreggatedMetrics(
        mean=float(np.mean(agg_metrics)),
        std=float(np.std(agg_metrics)),
        max=np.max(agg_metrics),
        min=np.min(agg_metrics)
    )
