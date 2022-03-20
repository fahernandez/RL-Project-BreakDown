"""
Experiment 3
Description: CCN with one dense layer.
-- Not punishing the agent by losing lives with -1 reward
-- With normalized rewards
"""
from src.model import PolicyNetwork


class ModelExperiment3(PolicyNetwork):
    NAME = '3'

    def __init__(self, execution_number):
        super(ModelExperiment3, self).__init__(self.NAME, execution_number)
        self.punish_agent = False
        self.normalized_rewards = True

