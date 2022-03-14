"""
Program to train an agent to play Atari Pong
Solution based on:
    1. https://github.com/gameofdimension/policy-gradient-pong/blob/master/policy_gradient_pong.py
    2. https://github.com/omkarv/pong-from-pixels/blob/master/pong-from-pixels.py
    3. https://medium.com/swlh/policy-gradient-reinforcement-learning-with-keras-57ca6ed32555
"""
import numpy as np
import gym
import gc


class BreakoutGame:
    # Action space
    NOOP = 0
    FIRE = 1
    MOVE_RIGHT = 2
    MOVE_LEFT = 3
    ACTION_SPACE = np.array([NOOP, FIRE, MOVE_RIGHT, MOVE_LEFT])  # Amount of available actions

    # Game configuration
    FRAME_SKIP = 5  # Each n frames the play screen will be sampled
    EPISODES = 10000  # Episodes to be played
    PIXELS_NUM = 5040

    def __init__(self, policy_network, resume, render_type):
        # Initialize variables
        self.__game_frames = []
        self.__gradients = []
        self.__action_probabilities = []
        self.__rewards = []

        # Initialize the game
        self.__env = gym.make(
            'Breakout-v0',
            obs_type='rgb',             # ram | rgb | grayscale/Observation return type
            frameskip=self.FRAME_SKIP,  # frame skip/Amount of frames to wait for getting a frame sample
            render_mode=render_type)    # None | human | rgb_array

        self.__env = self.__env.unwrapped

        # Policy gradient has high variance, seed for reproducability
        self.__env.seed(1)

        print("env.action_space", self.__env.action_space)
        print("env.meaning", self.__env.get_action_meanings())

        # Initialize the NN model
        self.__policy_network = policy_network
        self.__policy_network.set_action_space(self.ACTION_SPACE)
        self.__policy_network.load(resume)

        # Reset the game prior starting (This will clean prior states)
        self.__reset()

    def play(self):
        prev_game_frame = np.zeros((self.PIXELS_NUM,))  # used in computing the difference frame

        # A clean episode starts at zero so the first episode will be +1
        # We save completed episodes so the next episode will be +1
        for i_episode in range(self.__policy_network.get_episode()+1, self.EPISODES):
            done = False

            # Restart the game and kick the ball
            self.__env.reset()
            # Action 1 is fire on the Breakout game
            game_frame, _, _, game_state = self.__env.step(1)
            # We are getting negatives rewards for each time the agent lost a ball
            actual_lives = game_state['lives']

            # Each episode is made of n points
            while not done:
                # 1. Preprocess the Image to reduce noise and dimensionality
                game_frame, prev_game_frame = self.__pre_process_image(game_frame, prev_game_frame)

                # 2. Produce an action following the Policy function
                action, action_probability = self.__policy_network.produce_action(game_frame)

                # 3. Interact with the environment by executing the action
                new_game_frame, reward, done, game_state = self.__env.step(action)

                # 4. Add a negative reward by losing lives (In case it happened)
                new_lives = game_state['lives']
                if actual_lives != new_lives:
                    reward = -1
                actual_lives = new_lives

                # 5. Store the state, value, action and reward taken in this step
                self.__record_dynamics(game_frame, action, action_probability, reward)

                # Update the actual game frame
                game_frame = new_game_frame

                # Delete variables (there is a memory leak problem, I'm exploring causes)
                del new_game_frame
                del reward
                del new_lives

            # 5. Update the policy after the episodes finishes
            self.__policy_network.update_policy(
                self.__gradients,
                self.__game_frames,
                self.__action_probabilities,
                self.__rewards,
                i_episode)

            # 6. Reset the Game dynamics for the next episode
            self.__reset()

            # 7. Call the garbage collector to reduce memory consumption
            # https://groups.google.com/g/h5py/c/_a35vzQzRrg?pli=1
            gc.collect()

    def __record_dynamics(self, game_frame, action, action_probability, reward):
        """
        Store the sequence of state, actions and values for the actual episode

        :param game_frame: Game state
        :param action: Action taken
        :param action_probability: Action probabilities
        :param reward: Reward perceived by the agend interaction
        """
        encoded_action = self.__hot_encode_action(action)
        self.__gradients.append(encoded_action-action_probability)
        self.__game_frames.append(game_frame)
        self.__rewards.append(reward)
        self.__action_probabilities.append(action_probability)

        # Delete variables (there is a memory leak problem, I'm exploring causes)
        del encoded_action

    def __reset(self):
        """ Reset the game dynamic after each episode"""
        # Delete variables (there is a memory leak problem, I'm exploring causes)
        del self.__game_frames
        del self.__gradients
        del self.__action_probabilities
        del self.__rewards

        self.__game_frames = []
        self.__gradients = []
        self.__action_probabilities = []
        self.__rewards = []

    def __hot_encode_action(self, action):
        """
        Encoding the actions into a binary list where only the action taken has a value.
        The policy network has two outputs so this weights just the output of the action executed

        :param: Encode the action to use them while updating the weights
        """

        action_encoded = np.zeros(len(self.ACTION_SPACE), np.float32)
        if action == self.NOOP:
            action_encoded[0] = 1

        if action == self.FIRE:
            action_encoded[1] = 1

        if action == self.MOVE_RIGHT:
            action_encoded[2] = 1

        if action == self.MOVE_LEFT:
            action_encoded[3] = 1

        return action_encoded

    def __pre_process_image(self, game_frame, prev_game_frame):
        """
        Transform the original image game image into a less noise representation
        This will make easier the ball recognizing task

        :param game_frame: 210x160x3 RGB image
        :param prev_game_frame: Previous image free
        :return: 6000 (75x80) 1D float vector
        """
        # Convert Image to gray scale (Matlab formula)
        r, g, b = game_frame[:, :, 0], game_frame[:, :, 1], game_frame[:, :, 2]
        game_frame = 0.2989 * r + 0.5870 * g + 0.1140 * b
        game_frame = np.around(game_frame)

        # crop to reduce redundant parts of image (i.e. after ball passes paddle)
        game_frame = game_frame[55:195, 8:151]  # Don't cut anything in Y. Y completely is where the game happends
        game_frame = game_frame[::2, ::2]  # down sample the image
        game_frame[game_frame == 142.0] = 0  # erase background 1
        game_frame[game_frame == 127.0] = 0  # erase background 2

        # plt.imshow(game_frame, interpolation='nearest')
        # plt.show()
        current_game_frame = game_frame.astype(float).ravel()

        # This make the process complete MDP
        # This trick allows discarding previous game history because we observe movement
        game_frame = current_game_frame - prev_game_frame

        # Game validation safe kipping
        assert game_frame.size == self.PIXELS_NUM

        return game_frame, current_game_frame