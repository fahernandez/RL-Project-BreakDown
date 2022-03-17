"""
Main experiment execution program.
Experiments made:
1.
"""

# https://colab.research.google.com/drive/1BTiGjP_FD0PdYfazpn61nWPC0YJuZPOE?usp=sharing#scrollTo=u-WL_FAE1hI0
# https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
# http://karpathy.github.io/2016/05/31/rl/
# https://jair.org/index.php/jair/article/view/11182/26388
# https://arxiv.org/pdf/1807.08452.pdf
# https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
# https://lilianweng.github.io/posts/2018-04-08-policy-gradient/

import tensorflow as tf
from src.experiments.experiment_1 import ModelExperiment1
from src.experiments.experiment_2 import ModelExperiment2
from src.experiments.experiment_3 import ModelExperiment3
from src.experiments.experiment_4 import ModelExperiment4
from src.game import BreakoutGame
import logging
import threading

# Verify GPU devices
local_device_protos = tf.config.list_physical_devices('GPU')
if len(local_device_protos):
    for device in local_device_protos:
        print("Executing with device type {}, name {}".format(device.device_type, device.name))

RESUME = False  # Resume execution from last point
RENDER_TYPE = 'rgb_array'  # None | human | rgb_array

if __name__ == "__main__":
    # threads = list()
    # experiments = {
    #     "1": ModelExperiment1,
    #     "2": ModelExperiment2,
    #     "3": ModelExperiment3,
    #     "4": ModelExperiment4,
    #
    # }
    # experiment_repetition = 1
    #
    # def thread_function(experiment, rep):
    #     base_experiment = experiment(rep)
    #     BreakoutGame(base_experiment, RESUME, RENDER_TYPE).play()

    # for num, exp in experiments.items():
    #     # +1 because the count doesn't start at zero
    #     for repetition in range(1, experiment_repetition+1):
    #         logging.info("Main: create and start thread for exp {} and exec {}.", num, repetition)
    #         x = threading.Thread(target=thread_function, args=(exp, repetition))
    #         threads.append(x)
    #         x.start()
    #
    # for index, thread in enumerate(threads):
    #     logging.info("Main: before joining thread %d.", index)
    #     thread.join()
    #     logging.info("Main : thread %d done", index)

    base_experiment = ModelExperiment3(1)
    BreakoutGame(base_experiment, RESUME, RENDER_TYPE).play()
