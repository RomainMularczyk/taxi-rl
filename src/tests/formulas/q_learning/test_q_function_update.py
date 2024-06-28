from lib.formulas.q_learning import QLearning
from lib.policies.RandomSamplePolicy import RandomSamplePolicy


def test_q_learning_q_function_update():
    q_learning = QLearning(
        initial_seed=0,
        cutoff_score=0,
        observation_space=5,
        action_space=4,
        policy=RandomSamplePolicy(),
        gamma=1.0
    )
