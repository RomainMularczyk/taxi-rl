import gymnasium as gym
from lib.models.Action import Action, ActionWithReward
from lib.policies.EpsilonGreedyPolicy import EpsilonGreedyPolicy


def test_epsilon_greedy_policy_explore_legal():
    env = gym.make("Taxi-v3")
    policy = EpsilonGreedyPolicy(
        game_env=env,  # type: ignore
        seed=42,
        legal=True
    )
    policy.game_env.back_to(366)
    result = policy._explore()
    expected = [
        ActionWithReward(
            action=Action.SOUTH,
            reward=-1.0,
            probability=float(1/3)
        ),
        ActionWithReward(
            action=Action.NORTH,
            reward=-1.0,
            probability=float(1/3)
        ),
        ActionWithReward(
            action=Action.WEST,
            reward=-1.0,
            probability=float(1/3)
        )
    ]
    assert any(result == exp for exp in expected)


def test_epsilon_greedy_policy_explore_random():
    env = gym.make("Taxi-v3")
    policy = EpsilonGreedyPolicy(
        game_env=env,  # type: ignore
        seed=42,
        legal=False
    )
    result = policy._explore()
    expected = [
        ActionWithReward(
            action=Action.SOUTH,
            reward=-1.0,
            probability=float(1/6)
        ),
        ActionWithReward(
            action=Action.NORTH,
            reward=-1.0,
            probability=float(1/6)
        ),
        ActionWithReward(
            action=Action.EAST,
            reward=-1.0,
            probability=float(1/6)
        ),
        ActionWithReward(
            action=Action.WEST,
            reward=-1.0,
            probability=float(1/6)
        ),
        ActionWithReward(
            action=Action.PICK_UP,
            reward=-10.0,
            probability=float(1/6)
        ),
        ActionWithReward(
            action=Action.DROP_OFF,
            reward=-10.0,
            probability=float(1/6)
        )
    ]
    assert any(result == exp for exp in expected)


def test_epsilon_greedy_policy_tradeoff():
    env = gym.make("Taxi-v3")
    policy = EpsilonGreedyPolicy(
        game_env=env,
        seed=42,
        epsilon=1.0
    )
    policy._epsilon_tradeoff(random=1.0)


def test_epsilon_greedy_policy_1_epsilon_next_action():
    env = gym.make("Taxi-v3")
    policy = EpsilonGreedyPolicy(
        game_env=env,
        seed=42,
        epsilon=1.0
    )

def test_epsilon_greedy_policy_0_epsilon_next_action():
    env = gym.make("Taxi-v3")
    policy = EpsilonGreedyPolicy(
        game_env=env,
        seed=42,
        epsilon=0.0
    )

def test_epsilon_greedy_policy_09_epsilon_next_action():
    env = gym.make("Taxi-v3")
    policy = EpsilonGreedyPolicy(game_env=env, seed=42)
    pass

def test_epsilon_greedy_policy_1_epsilon_01_decay_next_action():
    env = gym.make("Taxi-v3")
    policy = EpsilonGreedyPolicy(game_env=env, seed=42)
    pass

def test_epsilon_greedy_policy_1_epsilon_05_decay_next_action():
    env = gym.make("Taxi-v3")
    policy = EpsilonGreedyPolicy(game_env=env, seed=42)
    pass

def test_epsilon_greedy_policy_09_epsilon_01_decay_next_action():
    env = gym.make("Taxi-v3")
    policy = EpsilonGreedyPolicy(game_env=env, seed=42)
    pass

def test_epsilon_greedy_policy_09_epsilon_05_decay_next_action():
    env = gym.make("Taxi-v3")
    policy = EpsilonGreedyPolicy(game_env=env, seed=42)
    pass
