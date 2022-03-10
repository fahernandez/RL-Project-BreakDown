"""
Program to train an agent to play Atari Pong
Solution based on:
    1. https://github.com/gameofdimension/policy-gradient-pong/blob/master/policy_gradient_pong.py
    2. https://github.com/omkarv/pong-from-pixels/blob/master/pong-from-pixels.py
    3. https://medium.com/swlh/policy-gradient-reinforcement-learning-with-keras-57ca6ed32555
"""
import numpy as np
import gym
import tensorflow as tf

            # from matplotlib import pyplot as plt
            # plt.imshow(game_frame, interpolation='nearest')
            # plt.show()

# ################## Tuning parameters ###########
# Start
REWARD_DISCOUNT_FACTOR_GAMMA = 0.99  # discount factor for reward
LR = 1e-2  # Weight Learning Rate
ALPHA_LR = 1e-4 # Learning rate of the policy gradient. This control how each update will alter the true gradients.
L1_REG = 1e-5
L2_REG = 1e-4
DECAY_RATE = 0.99  # decay factor for RMSProp leaky sum of grad^2
PIXELS_NUM = 5475
HIDDEN_UNITS = 100
BATCH_SIZE = 1   # Batch size for mini-batch gradient decent
EPISODE_POINTS = 3  # Each episode is defined by a fix amount of points.
FRAME_SKIP = 5  # Each n frames the play screen will be sampled
EPISODES = 10000  # Episodes to be played
RENDER_TYPE = 'rgb_array'  # None | human | rgb_array
MOVE_DOWN = 3
MOVE_UP = 2
ACTION_SPACE = np.array([MOVE_UP, MOVE_DOWN])  # Amount of available actions
# End


class PongGame:
    def __init__(self):
        # Initialize the game
        self._env = gym.make(
            'Pong-v0',
            obs_type='rgb',           # ram | rgb | grayscale/Observation return type
            frameskip=FRAME_SKIP,     # frame skip/Amount of frames to wait for getting a frame sample
            render_mode=RENDER_TYPE)  # None | human | rgb_array//Don't need to be modified

        self._policy_network = PolicyNetwork()
        self._total_rewards = []
        self._reset()

    def play(self):
        prev_game_frame = np.zeros((PIXELS_NUM,))  # used in computing the difference frame

        # Play
        game_frame = self._env.reset()
        for i_episode in range(EPISODES):
            points = 0  # Points in the episode

            # Each episode is made of n points
            while EPISODE_POINTS > points:
                # 1. Preprocess the Image to reduce noise and dimensionality
                game_frame, prev_game_frame = self._pre_process_image(game_frame, prev_game_frame)

                # 2. Produce an action following the Policy Network
                action, action_probability = self._policy_network.produce_action(game_frame)

                # 3. Interact with the environment by executing the action
                new_game_frame, reward, done, _ = self._env.step(action)

                # 4. Store the state, value, action and reward taken in this step
                self._record_dynamics(game_frame, action, action_probability, reward)

                # Keep track of the amount of points scored
                points += abs(reward)

                # Update the game frame
                game_frame = new_game_frame

            # For each completed episode
            print('Episode {}'.format(i_episode+1))
            self._total_rewards.append(points)

            # 5. Update the policy after the episodes finishes-Use Mini batches updates
            if i_episode % BATCH_SIZE == 0:
                print('Updating weights on episode {}'.format(i_episode))
                self._update_policy()

            # 6. Reset the Game for the next episode
            game_frame = self._env.reset()

    def _update_policy(self):
        """Updates the policy network using the NN model.
        This function is used after the MC sampling is done - following
        delta/theta = alpha * gradient + log pi
        """

        # get X
        game_frames = np.vstack(self._game_frames)

        # get Y
        gradients = np.vstack(self._gradients)
        rewards = np.vstack(self._rewards)
        discounted_rewards = self._get_discount_rewards(rewards)
        gradients *= discounted_rewards
        # This is the policy correction based on the environment feedback
        gradients = ALPHA_LR*np.vstack([gradients])+self._action_probabilities

        history = self._policy_network.get_model().train_on_batch(game_frames, gradients)

        self._reset()

        return history

    def _record_dynamics(self, game_frame, action, action_probability, reward):
        """
        Store the sequence of state, actions and values for the actual episode

        :param game_frame: Game state
        :param action: Action taken
        :param action_probability: Action probabilities
        :param reward: Reward perceived by the agend interaction
        :return:
        """
        encoded_action = self._hot_encode_action(action)
        self._gradients.append(encoded_action-action_probability)
        self._game_frames.append(game_frame)
        self._rewards.append(reward)
        self._action_probabilities.append(action_probability)

    def _reset(self):
        self._game_frames = []
        self._gradients = []
        self._rewards = []
        self._action_probabilities = []

    @staticmethod
    def _hot_encode_action(action):
        """
        Encoding the actions into a binary list where only the action taken has a value.
        The policy network has two outputs so this weights just the output of the action executed

        :return:
        """

        action_encoded = np.zeros(len(ACTION_SPACE), np.float32)
        if action == MOVE_UP:
            action_encoded[0] = 1

        if action == MOVE_DOWN:
            action_encoded[1] = 1

        return action_encoded

    @staticmethod
    def _pre_process_image(game_frame, prev_game_frame):
        """
        Transform the original image game image into a less noise representation
        This will make easier the ball recognizing task

        :param game_frame: 210x160x3 RGB image
        :param prev_game_frame: Previous image free
        :return: 6000 (75x80) 1D float vector
        """
        # Convert Image to gray scale (Matlab formula)
        from matplotlib import pyplot as plt

        r, g, b = game_frame[:, :, 0], game_frame[:, :, 1], game_frame[:, :, 2]
        game_frame = 0.2989 * r + 0.5870 * g + 0.1140 * b

        # crop to reduce redundant parts of image (i.e. after ball passes paddle)
        game_frame = game_frame[34:183, 0:146]
        game_frame = game_frame[::2, ::2]  # down sample the image
        game_frame[game_frame == 106.74810] = 0  # erase background 1
        game_frame[game_frame == 87.24360] = 0  # erase background 1

        game_frame[game_frame != 0] = 1  # everything else (paddles, ball) just set to 1.

        current_game_frame = game_frame.astype(float).ravel()

        # This make the process complete MDP
        # This trick allows discarding previous game history because we observe movement
        game_frame = current_game_frame - prev_game_frame

        # Game validation safe kipping
        assert game_frame.size == PIXELS_NUM

        return game_frame, current_game_frame

    @staticmethod
    def _get_discount_rewards(reward):
        """
        Take 1D float array of rewards and compute discounted reward.
        The discounted rewards are used to weight the state actions in the episode

        :param reward:
        :return: Sequence of discounted rewards per state and action
        """
        discounted_reward = np.zeros_like(reward)
        running_add = 0
        for t in reversed(range(0, len(reward))):
            if reward[t] != 0:
                running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * REWARD_DISCOUNT_FACTOR_GAMMA + reward[t]
            discounted_reward[t] = running_add

        # normalize discounted rewards
        mean_rewards = np.mean(discounted_reward)
        std_rewards = np.std(discounted_reward)
        norm_discounted_rewards = (
            discounted_reward - mean_rewards)/(std_rewards+1e-7)  # avoiding zero div

        return norm_discounted_rewards


class PolicyNetwork(tf.keras.Model):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        # Define the Network Arquitecture
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                HIDDEN_UNITS,
                input_shape=(None, PIXELS_NUM),
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1_REG, l2=L2_REG),
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            ),
            tf.keras.layers.Dense(
                len(ACTION_SPACE),
                activation="softmax",
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=L1_REG, l2=L2_REG),
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            )
        ])

        # output shape is according to the number of action
        # The softmax function outputs a probability distribution over the actions
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.RMSprop(LR, decay=DECAY_RATE)
        )

    def get_model(self):
        return self.model

    def produce_action(self, game_image_frame):
        """
        Samples the next action based on the policy probability distribution of the actions
        :param game_image_frame:
        :return: action to execute, probability distribution for all the actions
        """
        # get action probably
        action_probability = self.model.predict(
            game_image_frame.reshape((-1, game_image_frame.size))
        ).flatten()

        # norm action probability distribution
        action_probability /= np.sum(action_probability)

        # sample action based on the Network probability
        action = np.random.choice(
            ACTION_SPACE,
            1,
            p=action_probability
        )[0]

        return action, action_probability


PongGame().play()
