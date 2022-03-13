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
from src.game import BreakoutGame

# Verify GPU devices
local_device_protos = tf.config.list_physical_devices('GPU')
if len(local_device_protos):
    for device in local_device_protos:
        print("Executing with device type {}, name {}".format(device.device_type, device.name))

RESUME = False  # Resume execution from last point
RENDER_TYPE = 'human'  # None | human | rgb_array

exp1 = ModelExperiment1(1)
BreakoutGame(exp1, RESUME, RENDER_TYPE).play()
