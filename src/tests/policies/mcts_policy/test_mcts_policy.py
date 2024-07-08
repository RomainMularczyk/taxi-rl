import gymnasium as gym
import numpy as np
from lib.data.monte_carlo_tree import MonteCarloTree
from lib.models.EnvironmentInfo import EnvironmentInfo
from lib.models.Node import Node
from lib.models.Action import Action
from lib.policies.MonteCarloPolicy import MonteCarloPolicy
from lib.environment.environment import GameEnvironment


def test_no_winning_node_found():
    env = gym.make('Taxi-v3', render_mode='ansi')
    game_env = GameEnvironment(env, seed=4) # Win at 8 of depth
    actions = [a for a in Action if a is not Action.DROP_OFF] # All possible actions
    depth = 2 # Depth that can not allow the training to find winning Node
    MC_policy = MonteCarloPolicy(
        game_env=game_env,
        actions=actions,
        depth=depth
    )
    MC_policy.train_pickup()
    result = MC_policy.tree.deepest_layer_nodes
    expected_deepest_layer_nodes = [
        Node(
            action=Action.SOUTH,
            depth=depth,
            parent=None,
            state=468,
            env_info=EnvironmentInfo(**{
                "prob": 1.0,
                "action_mask": np.array([0, 1, 1, 0, 0, 0], dtype='int8') # type: ignore
            }),
            path='2A'
        ),
        Node(
            action=Action.NORTH,
            depth=depth,
            parent=None,
            state=268,
            env_info=EnvironmentInfo(**{
                "prob": 1.0,
                "action_mask": np.array([1, 1, 1, 1, 0, 0], dtype='int8') # type: ignore
            }),
            path='2B'
        ),
        Node(
            action=Action.EAST,
            depth=depth,
            parent=None,
            state=388,
            env_info=EnvironmentInfo(**{
                "prob": 1.0,
                "action_mask": np.array([1, 1, 0, 1, 0, 0], dtype='int8') # type: ignore
            }),
            path='2C'
        ),
        Node(
            action=Action.NORTH,
            depth=depth,
            parent=None,
            state=388,
            env_info=EnvironmentInfo(**{
                "prob": 1.0,
                "action_mask": np.array([1, 1, 0, 1, 0, 0], dtype='int8') # type: ignore
            }),
            path='2B'
        ),
        Node(
            action=Action.WEST,
            depth=depth,
            parent=None,
            state=468,
            env_info=EnvironmentInfo(**{
                "prob": 1.0,
                "action_mask": np.array([0, 1, 1, 0, 0, 0], dtype='int8') # type: ignore
            }),
            path='2D',
        ),
    ]
    for i, n in enumerate(expected_deepest_layer_nodes):
        n.reward = -1
        n.cumul_reward = -2
        assert n.action == result[i].action
        assert n.depth == result[i].depth
        # assert n.parent == result[i].parent
        assert n.state == result[i].state
        assert (n.env_info.action_mask == result[i].env_info.action_mask).all()
        assert n.path == result[i].path
        assert n.reward == result[i].reward
        assert n.cumul_reward == result[i].cumul_reward