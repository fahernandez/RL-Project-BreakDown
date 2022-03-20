"""
Experiment 1
Description: CCN with one dense layer.
-- Not punishing the agent by losing lives with -1 reward
-- Without normalized rewards
"""
from src.model import PolicyNetwork


class ModelExperiment1(PolicyNetwork):
    NAME = '1'

    def __init__(self, execution_number):
        super(ModelExperiment1, self).__init__(self.NAME, execution_number)
        self.punish_agent = False
        self.normalized_rewards = False
