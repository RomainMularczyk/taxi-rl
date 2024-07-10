import logging
import random
from typing import Tuple
from gymnasium.wrappers.time_limit import TimeLimit as GymnasiumGameEnvironment
from lib.data.monte_carlo_tree import MonteCarloTree
from lib.policies.Policy import Policy
from lib.models.Action import Action, ActionWithReward
from lib.models.GameStatus import GameStatus
from lib.models.Stage import Stage
from lib.models.Node import Node
from lib.models.EnvironmentInfo import EnvironmentInfo
from lib.errors.GameAlreadyWonException import GameAlreadyWonException
from lib.logs.logger import get_logger


LOGGER = get_logger()
LOGGER.setLevel(logging.ERROR)


class MonteCarloPolicy(Policy):
    """
    This policy will choose action based on the MonteCarloTree built
    after giving 2 parameters :
    - actions: a list [int] of all actions the agent can make
    - depth: the depth of the tree, meaning the maximum amount of steps
    the agent might have to take to complete pickup/dropoff
    """

    def __init__(
        self,
        game_env: GymnasiumGameEnvironment,
        depth: int,
        seed: int | None = None
    ):
        super().__init__(game_env=game_env, seed=seed)
        self.seed = seed
        self.depth = depth
        self.actions: list[Action] = list(Action)

        root_node = Node(
            depth=0,
            state=self.game_env.initial_state,
            env_info=self.game_env.initial_info
        )
        self.current_node: Node = root_node
        self.current_stage: Stage = Stage.PICK

        self.pick_tree: MonteCarloTree = self.generate_tree(
            root_node=root_node,
            depth=depth,
        )
        self.drop_tree: MonteCarloTree | None = None

    def reset_hyperparameters(self, reset_env: bool = False) -> None:
        """
        Resets the environment and the hyperparameters of the policy.

        Parameters
        ----------
        reset_env: bool, default=False
            Decide if the environment should be also reset.
        """
        if reset_env:
            self.__init__(self.game_env.env, depth=self.depth, seed=self.seed)

    def possible_actions(self) -> Tuple[Action, ...]:
        return Action.legal_actions(self.game_env.info.action_mask)

    def next_action(self, render: bool = False) -> ActionWithReward:
        """
        Choose the next action to take after having built a Tree
        of `self.depth`.
        `next_action`will find the `best_node` and then return its
        `first_layer_parent`.

        Returns
        -------
        action_with_reward: ActionWithReward
            The ActionWithReward representing the next action to take.
        """
        if self.game_env.passenger_droppedoff(self.current_node.state):
            raise GameAlreadyWonException(
                "You can't call next_action() cause the game has finished,"
                " the passenger is succeffully dropped off."
            )
        best_node = None
        self.current_node.children = []
        self.current_node.depth = 0
        self.current_node.path = ''
        self.current_node.parent = None
        self.current_node.reward = 0.0
        self.current_node.cumul_reward = 0.0
        # Call the training function
        if self.current_stage is Stage.PICK:
            self.pick_tree = self.generate_tree(
                root_node=self.current_node,
                depth=self.depth
            )
            self.train_pickup()
            if self.pick_tree.winning_node is not None:
                LOGGER.info(
                    "Found PICK stage winning node:"
                    f" {self.pick_tree.winning_node}"
                )
                best_node = self.pick_tree.winning_node
            else:
                max_cumul_reward = max(
                    node.cumul_reward for node in
                    self.pick_tree.deepest_layer_nodes
                )
                best_nodes = [
                    node for node in self.pick_tree.deepest_layer_nodes
                    if node.cumul_reward == max_cumul_reward
                ]
                best_node = random.choice(best_nodes)
                # LOGGER.info(f"RANDOM best_node CHOSEN: {best_node}")
        elif self.current_stage is Stage.DROP:
            self.drop_tree = self.generate_tree(
                root_node=self.current_node,
                depth=self.depth,
            )
            self.train_dropoff()
            if self.drop_tree.winning_node is not None:
                best_node = self.drop_tree.winning_node
            else:
                max_cumul_reward = max(
                    node.cumul_reward for node
                    in self.drop_tree.deepest_layer_nodes
                )
                best_nodes = [
                    node for node in self.drop_tree.deepest_layer_nodes
                    if node.cumul_reward == max_cumul_reward
                ]
                best_node = random.choice(best_nodes)
        self.current_node = best_node.first_layer_parent
        self.game_env.back_to(self.current_node.state)
        if render:
            self.game_env.render()
        # Override the actual stage after training
        if (
            self.game_env.passenger_pickedup(self.current_node.state) is False
            and self.current_node.taxi_on_passenger is False
        ):
            self.current_stage = Stage.PICK
        else:
            self.current_stage = Stage.DROP

        if self.game_env.passenger_droppedoff(self.current_node.state):
            game_status = GameStatus.TERMINATED
        else:
            game_status = GameStatus.RUNNING

        return ActionWithReward(
            action=best_node.first_layer_parent.action,
            probability=1.0,
            reward=best_node.first_layer_parent.reward,
            game_status=game_status
        )

    def generate_tree(
        self,
        root_node: Node,
        depth: int,
    ) -> MonteCarloTree:
        """
        Generate a MonteCarloTree data structure for PICK or DROP Stages.

        Attributes
        ----------
        root_node: Node
            The generated Node for the root of the Monte Carlo Tree.
        depth: int
            The depth of the tree generation.

        Returns
        -------
        tree: MonteCarloTree
            The built tree of all possibilities: `actions^depth` possibilities.
        """
        actions: Action = []
        if self.current_stage is Stage.PICK:
            actions = [a for a in Action if a is not Action.DROP_OFF]
        elif self.current_stage is Stage.DROP:
            actions = [a for a in Action if a is not Action.PICK_UP]
        tree = MonteCarloTree(
            root_node=root_node,
            actions=actions,
            depth=depth
        )

        return tree

    def train_pickup(self) -> None:
        # Initialize BFS list with root children
        self.pick_tree.bfs += self.pick_tree.root_node.children

        # Itérer le tableau et exécuter l'action associée à chaque noeud
        for node in self.pick_tree.bfs:
            # Take back env to parent Node state
            self.game_env.back_to(node.parent.state)

            # Exec Node action
            state, reward, _, _, info = self.game_env.env.step(
                node.action.value
            )

            # Process possible Cutoffs
            # Moved wall & wrong pick/drop
            illegal_action = bool(
                node.parent.env_info.action_mask[node.action.value] == 0
            )
            # Passed on visited state
            moved_back = bool(state in self.pick_tree.state_history)
            # Update du Tree et du Node
            self.pick_tree.visited_node += 1
            self.pick_tree.state_history.add(state)
            node.update_reward(reward)
            node.state = state
            node.env_info = EnvironmentInfo(**info)
            node.update_children_with(node.env_info)

            # Stop if passenger picked up
            if self.game_env.passenger_pickedup(state):
                node.taxi_on_passenger = True
                LOGGER.info(f"Found PICK stage winning node : {node}")
                self.pick_tree.winning_node = node
                self.current_stage = Stage.DROP
                drop_root_node = Node(
                    depth=0,
                    action=node.action,
                    parent=node.parent,
                    state=node.state,
                    env_info=node.env_info,
                    children=[],
                    path=node.path,
                    reward=node.reward,
                    cumul_reward=node.cumul_reward
                )
                self.drop_tree = self.generate_tree(
                    root_node=drop_root_node,
                    depth=self.depth,
                )
                return

            # Cutoff si mauvaise action choisie
            if illegal_action is False and moved_back is False:
                self.pick_tree.bfs.extend(node.children)

            # Add Nodes of searching depth to list if no winning Node found
            if node.depth == self.depth:
                self.pick_tree.deepest_layer_nodes.append(node)

    def train_dropoff(self) -> None:
        # Initialize BFS list with root children
        self.drop_tree.bfs += self.drop_tree.root_node.children

        # Itérer le tableau et exécuter l'action associée à chaque noeud
        for node in self.drop_tree.bfs:

            # Take back env to parent Node state
            self.game_env.back_to(node.parent.state)

            # Exec Node action
            state, reward, _, _, info = self.game_env.env.step(
                node.action.value
            )

            # Process possible Cutoffs
            # Moved wall & wrong pick/drop
            illegal_action = bool(
                node.parent.env_info.action_mask[node.action.value] == 0
            )
            # Passed on visited state
            moved_back = bool(state in self.drop_tree.state_history)

            # Update du Tree et du Node
            self.drop_tree.visited_node += 1
            self.drop_tree.state_history.add(state)
            node.update_reward(reward)
            node.state = state
            node.env_info = EnvironmentInfo(**info)
            node.update_children_with(node.env_info)

            # Stop if passenger dropped off
            if self.game_env.passenger_droppedoff(state):
                LOGGER.info(f"Found DROP stage winning node : {node}")
                self.drop_tree.winning_node = node
                return

            # Cutoff if bad action chosen
            if illegal_action is False and moved_back is False:
                self.drop_tree.bfs.extend(node.children)

            # Add Nodes of searching depth to list if no winning Node found
            if node.depth is self.depth:
                self.drop_tree.deepest_layer_nodes.append(node)

    def train(self) -> None:
        """
        Full train with the MonteCarloPolicy.
        This will execute:
            - `train_pickup()`
            - `train_dropoff()`
        """
        self.train_pickup()
        self.train_dropoff()
