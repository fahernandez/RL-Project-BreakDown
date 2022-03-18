"""
Experiment 1
Description: CCN with one dense layer
"""
from src.model import PolicyNetwork
import tensorflow as tf


class ModelExperiment3(PolicyNetwork):
    NAME = '3'
    HIDDEN_UNITS = '100'

    def __init__(self, execution_number):
        super(ModelExperiment3, self).__init__(self.NAME, execution_number)

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
            tf.keras.layers.Reshape((80, 80, 1)),
            tf.keras.layers.Conv2D(
                filters=16,
                kernel_size=8,
                padding='same',
                activation='relu',
                strides=(4, 4),
            ),
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=4,
                padding='same',
                activation='relu',
                strides=(2, 2)
            ),
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=3,
                padding='same',
                activation='relu',
                strides=(1, 1)
            ),
            tf.keras.layers.Flatten(),
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
