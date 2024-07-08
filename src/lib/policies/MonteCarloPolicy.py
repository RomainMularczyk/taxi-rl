import random
from lib.data.monte_carlo_tree import MonteCarloTree
from lib.models.Action import Action
from lib.models.Stage import Stage
from lib.models.Node import Node
from lib.models.EnvironmentInfo import EnvironmentInfo
from lib.environment.environment import GameEnvironment
from lib.logs.logger import get_logger


LOGGER = get_logger()

class MonteCarloPolicy:
    """
    This policy will choose action based on the MonteCarloTree built after giving 2 parameters :
    actions: a list [int] of all actions the agent can make
    depth: the depth of the tree, meaning the maximum amount of steps the agent might have to take to complete pickup/dropoff
    """

    def __init__(
        self, 
        game_env: GameEnvironment, 
        depth: int
    ):
        self.game_env: GameEnvironment = game_env
        self.depth = depth
        self.actions: list[Action] = list(Action)

        root_node = Node(
            depth=0,
            state=game_env.initial_state,
            env_info=game_env.initial_info
        )

        self.pick_tree: MonteCarloTree = self.generate_tree(
            root_node=root_node,
            depth=depth,
            stage=Stage.PICK
        )
        self.drop_tree = None

    def next_action(
        self,
        root_node: Node,
        stage: Stage
    ) -> Node:
        """
        Choose the next action to take after having built a Tree of `self.depth`.
        `next_action`will find the `best_node` and then return its `first_layer_parent`.

        Returns
        -------
        node: Node
            The Node representing the next action to take.
        """
        best_node = None
        # Generate the tree
        self.pick_tree = self.generate_tree(
            root_node=root_node,
            depth=self.depth,
            stage=stage
        )
        # Call the training function
        if stage is Stage.PICK:
            self.train_pickup()
            if self.pick_tree.winning_node is not None:
                best_node = self.pick_tree.winning_node
            else:
                max_cumul_reward = max(node.cumul_reward for node in self.pick_tree.deepest_layer_nodes)
                best_nodes = [node for node in self.pick_tree.deepest_layer_nodes if node.cumul_reward == max_cumul_reward]
                best_node = random.choice(best_nodes)
        elif stage is Stage.DROP:
            self.train_dropoff()
            if self.drop_tree.winning_node is not None:
                best_node = self.drop_tree.winning_node
            else:
                max_cumul_reward = max(node.cumul_reward for node in self.drop_tree.deepest_layer_nodes)
                best_nodes = [node for node in self.drop_tree.deepest_layer_nodes if node.cumul_reward == max_cumul_reward]
                best_node = random.choice(best_nodes)
        
        return best_node.first_layer_parent

    def generate_tree(
        self,
        root_node: Node,
        depth: int,
        stage: Stage
    ) -> MonteCarloTree:
        """
        Generate a MonteCarloTree data structure for PICK or DROP Stages.

        Attributes
        ----------
        root_state: int
            The state for the root Node of the generated tree.
        root_info: EnvironmentInfo
            The info for the root Node of the generated tree.
        depth: int
            The depth of the tree generation.
        stage: Stage
            The Stage of the training : PICK or DROP searching Stage.
        
        Returns
        -------
        tree: MonteCarloTree
            The built tree of all possibilities: `actions^depth` possibilities.
        """
        actions: Action = []
        if stage is Stage.PICK:
            actions = [a for a in Action if a is not Action.DROP_OFF]
        elif stage is Stage.DROP:
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
            LOGGER.info("________________________________________________________________________")
            LOGGER.info(f"Reseting game env in the state of the parent Node ({node.parent.state})")
            self.game_env.back_to(node.parent.state)

            # Exec Node action
            LOGGER.info(f"Taking action {node.action}...")
            state, reward, _, _, info = self.game_env.env.step(node.action.value)
            LOGGER.info(f"Reward({reward}) Info({info})")
            LOGGER.info(self.game_env.env.render())

            # Process possible Cutoffs
            illegal_action: bool = bool(node.parent.env_info.action_mask[node.action.value] == 0) # Moved wall & wrong pick/drop
            moved_back: bool = bool(state in self.pick_tree.state_history) # Passed on visited state

            # Update du Tree et du Node
            self.pick_tree.visited_node += 1
            self.pick_tree.state_history.add(state)
            node.update_reward(reward)
            node.state = state
            node.env_info = EnvironmentInfo(**info)
            node.update_children_with(node.env_info)

            # Stop if passenger picked up
            if self.game_env.passenger_pickedup(state):
                LOGGER.info(f"Found winning node : {node.fullpath}")
                node.update_reward(float(10))
                self.pick_tree.winning_node = node
                self.drop_tree = self.generate_tree(
                    root_node=node,
                    depth=self.depth,
                    stage=Stage.DROP
                )
                return

            # Cutoff si mauvaise action choisie
            if illegal_action is False and moved_back is False:
                LOGGER.info(f"Adding {[node.path for node in node.children]} to BFS array.")
                self.pick_tree.bfs.extend(node.children)

            # Add Nodes of searching depth to list if no winning Node found
            if node.depth is self.depth:
                self.pick_tree.deepest_layer_nodes.append(node)

            LOGGER.debug(f"BFS array: {[node.path for node in self.pick_tree.bfs]}")

    def train_dropoff(self):
        # Initialize BFS list with root children
        self.drop_tree.bfs += self.drop_tree.root_node.children
