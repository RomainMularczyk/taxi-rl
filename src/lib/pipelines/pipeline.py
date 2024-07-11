import pandas as pd


def execute_pipeline(formula, policy, data):
    formula(policy, data)

def compute_stats() -> pd.DataFrame:
    total_reward, rewards_per_episode, total_steps, computing_time = execute_pipeline()
