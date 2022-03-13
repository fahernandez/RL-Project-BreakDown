"""
Program to train an agent to play Atari Pong
Solution based on:
    1. https://github.com/gameofdimension/policy-gradient-pong/blob/master/policy_gradient_pong.py
    2. https://github.com/omkarv/pong-from-pixels/blob/master/pong-from-pixels.py
"""
import numpy as np
import pickle
import gym
import tensorflow as tf
import time
import os

tf.compat.v1.disable_eager_execution()

# from tensorflow.python.client import device_lib
#
# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']
# get_available_gpus()

# ################## Tuning parameters ###########
# Start
GAMMA = 0.99 # discount factor for reward
LR = 1e-3 # Gradient step
DECAY_RATE = 0.99 # decay factor for RMSProp leaky sum of grad^2
PIXELS_NUM = 6000
HIDDEN_UNITS = 200
BATCH_SIZE = 10 # Batch size for mini-batch gradient decent
EPISODE_POINTS = 3 # Each episode is defined by a fix amount of points.
FRAME_SKIP = 10  # Each n frames the play screen will be sampled
EPISODES = 10000 # Episodes to be played
RESUME = False  # resume from previous checkpoint?
RENDER_TYPE = 'human' # None | human | rgb_array
# End


def pre_proces(image):
    """
    Transform the original image game image into a less noise representation
    This will make easier the ball recognization task

    :param image: 210x160x3 RGB image
    :return: 6000 (75x80) 1D float vector
    """
    image = image[35:185]  # crop to reduce redundant parts of image (i.e. after ball passes paddle)
    image = image[::2, ::2, 0]  # down sample the image
    image[image == 144] = 0  # erase background (background type 1)
    image[image == 109] = 0  # erase background (background type 2)
    image[image != 0] = 1  # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively
    return image.astype(float).ravel()  # ravel flattens an array and collapses it into a column vector


def discount_rewards(reward):
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
        running_add = running_add * GAMMA + reward[t]
        discounted_reward[t] = running_add
    return discounted_reward


def load_model(path):
    """
    Load the saved model
    :param path: file path to the model artefact
    :return:
    """
    model = pickle.load(open(path, 'rb'))
    return model['W1'].T, model['W2'].reshape((model['W2'].size, -1))


def make_network(pixels_num, hidden_units):
    """
    Define the Policy Neural Network model
    :param pixels_num: Length of input image after pre-processing
    :param hidden_units: Number of hidden units in Layer 1
    :return:
    """

    pixels = tf.compat.v1.placeholder(dtype=tf.compat.v1.float32, shape=(None, pixels_num))
    actions = tf.compat.v1.placeholder(dtype=tf.compat.v1.float32, shape=(None, 1))
    rewards = tf.compat.v1.placeholder(dtype=tf.compat.v1.float32, shape=(None, 1))

    with tf.compat.v1.variable_scope('policy'):
        hidden = tf.compat.v1.layers.dense(pixels,
                                           hidden_units,
                                           activation=tf.compat.v1.nn.relu,
                                           kernel_initializer=tf.keras.initializers.GlorotUniform())
        logits = tf.compat.v1.layers.dense(hidden,
                                           1,
                                           activation=None,
                                           kernel_initializer=tf.keras.initializers.GlorotUniform())

        out = tf.compat.v1.sigmoid(logits, name="sigmoid")
        cross_entropy = tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(labels=actions, logits=logits, name="cross_entropy")
        # This is the lost for policy gradient methods
        loss = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(rewards, cross_entropy, name="rewards"))

    opt = tf.compat.v1.train.RMSPropOptimizer(LR, decay=DECAY_RATE).minimize(loss)
    tf.compat.v1.summary.histogram("hidden_out", hidden)
    tf.compat.v1.summary.histogram("logits_out", logits)
    tf.compat.v1.summary.histogram("prob_out", out)
    merged = tf.compat.v1.summary.merge_all()

    return pixels, actions, rewards, out, opt, merged


def main():
    """
    Main game execution
    """
    reward_mean = -21

    tf.compat.v1.reset_default_graph()
    pix_ph, action_ph, reward_ph, out_sym, opt_sym, merged_sym = make_network(PIXELS_NUM, HIDDEN_UNITS)

    sess = tf.compat.v1.Session()
    saver = tf.compat.v1.train.Saver()
    writer = tf.compat.v1.summary.FileWriter('./log/train', sess.graph)

    if RESUME:
        saver.restore(sess, tf.compat.v1.train.latest_checkpoint('./log/checkpoints'))
    else:
        sess.run(tf.compat.v1.global_variables_initializer())

    # Start the game
    env = gym.make('Pong-v0',
        obs_type='rgb',                   # ram | rgb | grayscale/Observation return type/Don't need to be modified
        frameskip=FRAME_SKIP,             # frame skip/Amount of frames to wait for getting a frame sample
        render_mode=RENDER_TYPE           # None | human | rgb_array//Don't need to be modified
    )

    observation = env.reset()
    prev_frame = None # used in computing the difference frame
    xs, ys, ws, ep_ws, batch_ws = [], [], [], [], []

    # Restore from last episode
    last_episode = pickle.load(open('./log/step.p', 'rb')) if RESUME and os.path.exists('./log/step.p') else 0
    for i_episode in range(EPISODES - last_episode):
        points = 0 # Points in episode
        while EPISODE_POINTS > points:
            # Downsize and clean the original image
            current_frame = pre_proces(observation)
            # This make the process complete MDP
            # This trick allows discarding previous game history
            input_state = current_frame - prev_frame if prev_frame is not None else np.zeros((PIXELS_NUM,))
            prev_frame = current_frame

            # Safe kipping
            assert input_state.size == PIXELS_NUM

            # Feed forward the current state to calculate the next action
            tf_probs = sess.run(out_sym, feed_dict={pix_ph:input_state.reshape((-1, input_state.size))})

            # Make a random exploration
            y = 1 if np.random.uniform() < tf_probs[0,0] else 0
            action = 2 + y

            # Free memory
            del observation
            del current_frame

            # Interact with the environment by executing the action
            observation, reward, done, _ = env.step(action)

            # Save the state
            xs.append(input_state)
            # Save the action probability based on the state
            ys.append(y)
            # Save the reward
            ep_ws.append(reward)

            # Keep track of the amount of points scored
            points += abs(reward)

        # For each completed episode
        print('Episode {}'.format(i_episode))

        # Calculate the discounted rewards
        discounted_epr = discount_rewards(ep_ws)

        # Normalize the discounted rewards to reduce the varience of the weights
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        batch_ws += discounted_epr.tolist()

        # Hard process stop
        print(reward_mean)
        reward_mean = 0.99*reward_mean+(1-0.99)*(sum(ep_ws))
        rs_sum = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag="running_reward", simple_value=reward_mean)])
        writer.add_summary(rs_sum, global_step=i_episode)
        if reward_mean > 5.0:
            break

        del ep_ws
        ep_ws = []

        # MiniBatch update
        if i_episode % BATCH_SIZE == 0:
            exs = np.vstack(xs)
            eys = np.vstack(ys)
            ews = np.vstack(batch_ws)
            frame_size = len(xs)
            del xs
            del ys
            del discounted_epr
            del batch_ws

            stride = 20000
            pos = 0
            while True:
                end = frame_size if pos+stride>=frame_size else pos+stride
                batch_x = exs[pos:end]
                batch_y = eys[pos:end]
                batch_w = ews[pos:end]
                tf_opt, tf_summary = sess.run(
                    [opt_sym, merged_sym],
                    feed_dict={pix_ph:batch_x, action_ph:batch_y, reward_ph:batch_w}
                )
                pos = end
                if pos >= frame_size:
                    break
            # Free space
            xs = []
            ys = []
            batch_ws = []
            del exs
            del eys
            del ews
            del batch_x
            del batch_y
            del batch_w
        saver.save(sess, "./log/checkpoints/pg_{}.ckpt".format(i_episode))
        writer.add_summary(tf_summary, i_episode)
        print("datetime: {}, episode: {}, frame size: {}, reward: {}".\
                format(time.strftime('%X %x \%Z'), i_episode, frame_size, reward_mean))

        fp = open('./log/step.p', 'wb')
        pickle.dump(i_episode, fp)
        fp.close()

        observation = env.reset()


    env.close()

main()
