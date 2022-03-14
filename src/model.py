"""
Main Policy Gradient model
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.saving import hdf5_format
import h5py
import os


class PolicyNetwork(tf.keras.Model):
    # Neural network parameters-Less prompt to change (more standard)
    LR = 1e-2  # Weights Learning Rate
    L1_REG = 1e-4
    L2_REG = 1e-4
    DECAY_RATE = 0.99  # Decay factor for RMSProp leaky sum of grad^2

    # Experiment provided parameters-More prompt to change
    REWARD_DISCOUNT_FACTOR_GAMMA = 0.99  # discount factor for rewards following a trayectory
    ALPHA_LR = 1e-4  # Learning rate of the policy gradient. This control how each update will alter the true gradients.

    # Internal dynamics variables
    MODEL_STORAGE_NAME = 'latest'  # Name used to save the model after each weights updated
    MODEL_CHECKPOINT_AT = 10  # Defines how many episodes a version of the model will be stored

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
        self.__model = None
        self.__reward_discount_factor_gamma = self.REWARD_DISCOUNT_FACTOR_GAMMA
        self.__alpha_lr = self.ALPHA_LR

    def update_policy(self, gradients, game_frames, action_probabilities, rewards, episode):
        """
        Updates the policy network using the NN model.
        This function is used after the MC sampling is done - following
        delta/theta = alpha * gradient + log pi
        :param gradients: Prediction discrepancies
        :param game_frames: Game state
        :param action_probabilities: Action probabilities
        :param rewards: Reward perceived by the agent interaction
        :param episode: Episode executed
        """
        # Store the actual episode and the accumulated sum of rewards on the episode
        self.__episode = episode
        self.__total_rewards = np.append(self.__total_rewards, np.sum(rewards))

        print('Updating weights on exp {}, exec {}, episode {}, rewards {}, mean {}'.format(
            self.__name,
            self.__execution_number,
            self.__episode,
            self.__total_rewards[-1],
            np.mean(self.__total_rewards)
        ))

        # get X
        game_frames = np.vstack(game_frames)

        # get Y
        rewards = np.vstack(rewards)
        discounted_rewards = self.__get_discount_rewards(rewards)

        gradients = np.vstack(gradients)
        gradients *= discounted_rewards
        # This is the policy correction based on the environment feedback
        gradients = self.__alpha_lr*np.vstack([gradients]) + action_probabilities

        self.__model.train_on_batch(game_frames, gradients)

        # Save the model after each update
        self.__save(episode)

        # Delete variables (there is a memory leak problem, I'm exploring causes)
        del game_frames
        del gradients
        del discounted_rewards

    def produce_action(self, game_image_frame):
        """
        Samples the next action based on the policy probability distribution of the actions
        :param game_image_frame:
        :return: action to execute, probability distribution for all the actions
        """
        # get action probably
        # Because we are using a softmax output activation function, the output is already normalized
        action_probability = self.__model.predict(
            game_image_frame.reshape((-1, game_image_frame.size))
        ).flatten()

        # sample action based on the Network probability
        action = np.random.choice(self.__action_space, 1, p=action_probability)[0]

        return action, action_probability

    def __get_discount_rewards(self, reward):
        """
        Take 1D float array of rewards and compute discounted reward.
        The discounted rewards are used to weight the state actions in the episode

        :param reward:
        :return: Sequence of discounted rewards per state and action
        """
        discounted_reward = np.zeros_like(reward)
        cumulative = 0
        for t in reversed(range(0, len(reward))):
            # Reward -1 is given when a life is missed in the game
            if reward[t] == -1:
                cumulative = 0  # reset the sum, since this was a game boundary (the ball is out of the game)
            cumulative = cumulative * self.__reward_discount_factor_gamma + reward[t]
            discounted_reward[t] = cumulative

        # normalize discounted rewards
        mean_rewards = np.mean(discounted_reward)
        std_rewards = np.std(discounted_reward)
        norm_discounted_rewards = (
            discounted_reward - mean_rewards)/(std_rewards+1e-7)  # avoiding zero div

        # Delete variables (there is a memory leak problem, I'm exploring causes)
        del discounted_reward

        return norm_discounted_rewards

    def load(self, resume):
        pass

    def resume(self, resume):
        """Resume the model execution from last saved checkpoint"""
        if not resume:
            print("Model initialized from blank state")
            return False

        model_name = self.__get_store_path() + self.MODEL_STORAGE_NAME
        if not os.path.exists(model_name):
            print("Requested model {} doesn't exist. Model initialized from blank state".format(model_name))
            return False

        with h5py.File(model_name, mode='r') as f:
            self.__episode = f.attrs['episode']
            self.__total_rewards = f.attrs['total_rewards']
            self.__model = hdf5_format.load_model_from_hdf5(f)

            print("Loading Policy Network model {} trained on episode {}".format(
                model_name,
                self.__episode
            ))

            # Closed the file to free memory
            f.close()

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
            f.attrs['episode'] = self.__episode
            f.attrs['total_rewards'] = self.__total_rewards

            # Close the file to free memtory
            f.close()

        # Save the model by stages for latter review
        if episode % self.MODEL_CHECKPOINT_AT == 0 and episode != 0:
            model_name = path + str(episode)
            print("Saving mode checkpoint at episode {} and path".format(episode, model_name))
            with h5py.File(model_name, mode='w') as f:
                hdf5_format.save_model_to_hdf5(self.__model, f)
                f.attrs['episode'] = self.__episode
                f.attrs['total_rewards'] = self.__total_rewards

                # Close the file to free memory
                f.close()

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

    def set_model(self, model):
        self.__model = model
