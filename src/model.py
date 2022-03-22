"""
Main Policy Gradient model.
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.saving import hdf5_format
import tensorflow_probability as tfp
import h5py
import os


class PolicyNetwork(tf.keras.Model):
    # Neural network parameters-Less prompt to change (more standard)
    LR = 1e-4  # Weights Learning Rate
    L1_REG = 1e-4
    L2_REG = 1e-4
    HIDDEN_UNITS = '200'

    # Experiment provided parameters-More prompt to change
    REWARD_DISCOUNT_FACTOR_GAMMA = 0.99  # discount factor for rewards following a trajectory

    # Internal dynamics variables
    MODEL_STORAGE_NAME = 'latest.h5'  # Name used to save the model after each weights updated
    MODEL_CHECKPOINT_AT = 500  # Defines how many episodes a version of the model will be stored
    MODEL_TOTAL_REWARD_NAME = 'total_reward.npy'
    MODEL_TOTAL_POINT_NAME = 'total_point.npy'
    MODEL_EPISODE_NAME = 'episode.npy'
    MODEL_FAILOVER_CHECKPOINT = 20 # Save the model each n iterations to easily record

    def __init__(self, name, execution_number):
        """
        :param name: Experiment name
        :param execution_number: Experiment execution number
        """
        super(PolicyNetwork, self).__init__()
        self.__name = name
        self.__execution_number = execution_number
        self.__action_space = []
        self.__episode = 0
        self.__total_rewards = np.array([])
        self.__total_points = np.array([])
        self.__model = None

        # Punish the agent by loss lives
        self.punish_agent = True

        # Normalize the error attribution
        self.normalized_rewards = True

    def update_policy(self, state_memory, action_memory, reward_memory, point_memory, episode):
        """
        Updates the policy network using the NN model.
        This update is made following the Policy gradient Reinforce implementation
        :param state_memory: Game state memory
        :param action_memory: Action taken during each episode in a single trajectory
        :param reward_memory: Reward perceived by the agent interaction
        :param point_memory: Points received by the agent interaction
        :param episode: Episode executed
        """

        # Store the actual episode and the accumulated sum of rewards on the episode
        self.__episode = episode
        self.__total_rewards = np.append(self.__total_rewards, np.sum(reward_memory))
        self.__total_points = np.append(self.__total_points, np.sum(point_memory))

        print('Updating weights on exp {}, exec {}, episode {}, rewards {}, rewards mean {}, points {}, points mean {}'.format(
            self.__name,
            self.__execution_number,
            self.__episode,
            self.__total_rewards[-1],
            np.mean(self.__total_rewards[-100:]),
            self.__total_points[-1],
            np.mean(self.__total_points[-100:])
        ))

        # Convert the action and rewards to tensors
        actions = tf.convert_to_tensor(action_memory, dtype=tf.float32)
        rewards = np.array(reward_memory)

        # Get the discounted rewards
        discounted_rewards = self.__get_discount_rewards(rewards)

        # Calculate the lost function and make the gradient update
        with tf.GradientTape() as tape:
            loss = 0
            for idx, (g, state) in enumerate(zip(discounted_rewards, state_memory)):
                state = tf.convert_to_tensor([state], dtype=tf.float32)

                # Get the action probabilities as a tensor
                probs = self.__model(state)

                # Clipping 0/1 probabilities to avoid gradient update errors
                # We are calculating log(p(state)) and this will fail if 0/1
                probs = tf.clip_by_value(probs, clip_value_min=1e-8, clip_value_max=1-1e-8)

                # Get the log probability associated with that action
                action_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(actions[idx])

                # This is the lost function based on the REINFORCE algorithm
                loss += -g * tf.squeeze(log_prob)

        gradient = tape.gradient(loss, self.__model.trainable_variables)
        self.__model.optimizer.apply_gradients(zip(gradient, self.__model.trainable_variables))

        # Save the model after n updates
        if episode % self.MODEL_FAILOVER_CHECKPOINT == 0 and episode != 0:
            print('Saving model for fail-over recovery')
            self.__save(episode)

        # Delete variables (there is a memory leak problem, I'm exploring causes)
        del state_memory
        del action_memory
        del discounted_rewards
        del reward_memory
        del point_memory

    def produce_action(self, game_image_frame):
        """
        Samples the next action based on the policy probability distribution of the actions
        :param game_image_frame:
        :return: action to execute
        """
        # Feed forward the state into the NN to get the action probabilities
        probs = self.__model(
            game_image_frame.reshape((-1, game_image_frame.size))
        )

        # Create and sample an action based on the probability output of the NN
        # This allows the exploration fase on the algorithm
        action_probs = tfp.distributions.Categorical(probs=probs, validate_args=True)
        action = action_probs.sample()

        return action.numpy()[0]

    def set_model(self, resume, input_dim):
        """
        Load the Policy Gradient Network model for the experiment
        :param resume: Resume execution from last save checkpoint
        :param input_dim: Dimension of the input vector
        """
        # Resume the model execution from last checkpoint
        if self.__resume(resume):
            return

        # Load the Network Architecture
        self.__model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Reshape((80, 80, 1)),
            tf.keras.layers.Conv2D(
                filters=64,
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

        self.__model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.LR)
        )

        print(self.__model.summary())

    def __get_discount_rewards(self, rewards):
        """
        Take 1D float array of rewards and compute discounted reward.
        The discounted rewards are used to weight the state actions in the episode

        :param rewards:
        :return: Sequence of discounted rewards per state and action
        """
        discounted_reward = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            if rewards[t] != 0:
                running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * self.REWARD_DISCOUNT_FACTOR_GAMMA + rewards[t]
            discounted_reward[t] = running_add

        # normalize discounted rewards
        if self.normalized_rewards:
            # normalize discounted rewards
            mean_rewards = np.mean(discounted_reward)
            std_rewards = np.std(discounted_reward)
            discounted_reward = (
                discounted_reward - mean_rewards)/(std_rewards+1e-7)  # avoiding zero div

        return discounted_reward

    def __resume(self, resume):
        """Resume the model execution from last saved checkpoint"""
        if not resume:
            print("Model initialized from blank state")
            return False

        path = self.__get_store_path()
        model_name = path + self.MODEL_STORAGE_NAME
        if not os.path.exists(model_name):
            print("Requested model {} doesn't exist. Model initialized from blank state".format(model_name))
            return False

        with h5py.File(model_name, mode='r') as f:
            self.__model = hdf5_format.load_model_from_hdf5(f)

            # Closed the file to free memory
            f.close()

        self.__episode = np.load(path + self.MODEL_EPISODE_NAME)[0]
        self.__total_rewards = np.load(path + self.MODEL_TOTAL_REWARD_NAME)
        self.__total_points = np.load(path + self.MODEL_TOTAL_POINT_NAME)

        print("Loading Policy Network model {} trained on episode {}".format(
            model_name,
            self.__episode
        ))

        return self.__model is not None

    def __save(self, episode):
        """
        Save the model for fail over recovery
        :params: episode: each 2500 episodes a model will be saved as checkpoint for review
        """
        path = self.__get_store_path()
        if not os.path.exists(path):
            os.makedirs(path)

        model_name = path + self.MODEL_STORAGE_NAME
        with h5py.File(model_name, mode='w') as f:
            hdf5_format.save_model_to_hdf5(self.__model, f)

            # Close the file to free memory
            f.close()

        # Save the episode and the total reward achieved
        np.save(path + self.MODEL_EPISODE_NAME, np.array([self.__episode]))
        np.save(path + self.MODEL_TOTAL_REWARD_NAME, self.__total_rewards)
        np.save(path + self.MODEL_TOTAL_POINT_NAME, self.__total_points)

        # Save the model by stages for latter review
        if episode % self.MODEL_CHECKPOINT_AT == 0 and episode != 0:
            model_name = path + str(episode) + '.h5'
            print("Saving mode checkpoint at episode {} and path {}".format(episode, model_name))
            with h5py.File(model_name, mode='w') as f:
                hdf5_format.save_model_to_hdf5(self.__model, f)

                # Close the file to free memory
                f.close()

            # Save the rewards perceived to that moment
            np.save(
                path + str(episode) + '-' + self.MODEL_TOTAL_REWARD_NAME,
                self.__total_rewards
            )

            # Save the points perceived to that moment
            np.save(
                path + str(episode) + '-' + self.MODEL_TOTAL_POINT_NAME,
                self.__total_points
            )

    def __get_store_path(self):
        return './execution/exp_{}/exec_{}/'.format(self.__name, self.__execution_number)

    def get_episode(self):
        return self.__episode

    def set_action_space(self, action_space):
        self.__action_space = action_space

    def get_action_space(self):
        return self.__action_space

    def get_execution_number(self):
        return self.__execution_number
