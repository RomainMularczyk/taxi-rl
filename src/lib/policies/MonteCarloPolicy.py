from lib.data.monte_carlo_tree import MonteCarloTree
from lib.models.Action import Action
from lib.environment.environment import GameEnvironment


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
        self.game_env = game_env
        self.tree = MonteCarloTree(actions, depth)

    def train_pickup(self):
        # Initialiser le tableau de recherche BFS avec les enfants de '0' (root)
        BFS = self.tree.root_node.children
        # Itérer le tableau et exécuter l'action associée à chaque noeud
        for node in BFS:
            print(f"{node.action} is action {node.action.value}")
        # CutOff les branches où l'action est mauvaise
        # Ajouter tous les noeuds enfants de l'élément itéré au tableau

    def train_dropoff(self):
        pass