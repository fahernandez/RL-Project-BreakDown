"""
Experiment 1
Description: NN with 100 hidden units
"""
from src.model import PolicyNetwork
import tensorflow as tf


class ModelExperiment1(PolicyNetwork):
    # Nodes in the hidden layer
    HIDDEN_UNITS = 200
    NAME = '1'

    def __init__(self, execution_number):
        super(ModelExperiment1, self).__init__(self.NAME, execution_number)

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
                self.HIDDEN_UNITS,
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

        print(model.summary())

        self.set_model(model, self.NAME)
