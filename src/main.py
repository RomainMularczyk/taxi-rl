import argparse
import gymnasium as gym
from Logger import get_logger
from GameEnv import GameEnvironment
from lib.models.ParsedArgs import ParsedArgs

logger = get_logger()

parser = argparse.ArgumentParser(description="Taxi Driver RL script.")

# Script args with default values
parser.add_argument('-T', '--training', action='store_true', default=False, help='Define if the model is training (boolean)')
parser.add_argument('-e', '--episodes', type=int, default=1, help='Number of episodes to run')
parser.add_argument('-g', '--gamma', type=int, default=1, help='Gamma γ discount factor')

# Generate GameEnvironment
args: ParsedArgs = ParsedArgs(**vars(parser.parse_args()))
env = gym.make('Taxi-v3', render_mode='ansi')
game = GameEnvironment(env, args)
game.back_to(seed=302)

## Run pipeline
#result = pipeline.run(args)
#result.export()
