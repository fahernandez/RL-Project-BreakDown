"""
Experiment 1
Description: Description: NN with 100 hidden units followed by 50 hidden units
"""
from src.model import PolicyNetwork
import tensorflow as tf


class ModelExperiment2(PolicyNetwork):
    # Nodes in the hidden layer
    LAYER1_UNIT = 100
    LAYER2_UNIT = 50
    NAME = '2'

    def __init__(self, execution_number):
        super(ModelExperiment2, self).__init__(self.NAME, execution_number)

    def load(self, resume, input_dim):
        """
        Load the Policy Network model for the experiment
        :param resume: Resume execution from last save checkpoint
        :param input_dim: Dimension of the input vector
        """
        # Resume the model execution from last checkpoint
        if super().resume(resume):
            return

        # Load the Network Architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(
                self.LAYER1_UNIT,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.L1_REG, l2=self.L2_REG),
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            ),
            tf.keras.layers.Dense(
                self.LAYER2_UNIT,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.L1_REG, l2=self.L2_REG),
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            ),
            tf.keras.layers.Dense(
                len(self.get_action_space()),
                activation="softmax",
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=self.L1_REG, l2=self.L2_REG),
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            )
        ])

        self.set_model(model, self.NAME)

