"""
Experiment 2
Description: CCN with one dense layer.
-- Punishing the agent by losing lives with -1 reward
-- Without normalized rewards
"""
from src.model import PolicyNetwork


class ModelExperiment2(PolicyNetwork):
    NAME = '2'

    def __init__(self, execution_number):
        super(ModelExperiment2, self).__init__(self.NAME, execution_number)
        self.punish_agent = True
        self.normalized_rewards = False
