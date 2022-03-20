"""
Experiment 4
Description: CCN with one dense layer.
-- Punishing the agent by losing lives with -1 reward
-- With normalized rewards
"""
from src.model import PolicyNetwork


class ModelExperiment4(PolicyNetwork):
    NAME = '4'

    def __init__(self, execution_number):
        super(ModelExperiment4, self).__init__(self.NAME, execution_number)
        self.punish_agent = True
        self.normalized_rewards = True

