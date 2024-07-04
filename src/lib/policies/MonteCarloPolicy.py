from lib.data.monte_carlo_tree import MonteCarloTree
from lib.models.Action import Action
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
        actions: list[Action], 
        depth: int
    ):
        self.game_env: GameEnvironment = game_env
        self.tree = MonteCarloTree(
            root_state=game_env.initial_state,
            root_info=game_env.initial_info,
            actions=actions,
            depth=depth
        )

    def train_pickup(self):
        # Initialize BFS list with root children
        self.tree.bfs += self.tree.root_node.children
        # Itérer le tableau et exécuter l'action associée à chaque noeud
        for node in self.tree.bfs:
            # Take back env to parent Node state
            LOGGER.info("________________________________________________________________________")
            LOGGER.info(f"Reseting game env in the state of the parent Node ({node.parent.state})")
            self.game_env.back_to(node.parent.state)

            # Exec Node action
            LOGGER.info(f"Taking action {node.action}...")
            state, reward, _, _, info = self.game_env.env.step(node.action.value)
            LOGGER.info(self.game_env.env.render())  

            # Process possible Cutoffs
            illegal_action: bool = bool(node.parent.env_info.action_mask[node.action.value] == 0) # Moved wall & wrong pick/drop
            moved_back: bool = bool(state in self.tree.state_history) # TODO: Check side effects

            # Update du Tree et du Node
            self.tree.visited_node += 1
            self.tree.state_history.add(state)
            node.update_reward(reward)
            node.state = state
            node.env_info = EnvironmentInfo(**info)
            node.update_children_with(node.env_info)

            # Stop if passenger picked up
            if self.game_env.passenger_pickedup(state):
                LOGGER.info(f"Found winning node : {node.fullpath}")
                self.tree.picking_node = node
                return

            # Cutoff si mauvaise action choisie
            if illegal_action is False and moved_back is False:
                LOGGER.info(f"Adding {[node.path for node in node.children]} to BFS array.")
                self.tree.bfs.extend(node.children)

            LOGGER.debug(f"BFS array: {[node.path for node in self.tree.bfs]}")

    def train_dropoff(self):
        pass
