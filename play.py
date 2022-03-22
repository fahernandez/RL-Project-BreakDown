from src.game import BreakoutGame
from src.experiments.experiment_4 import ModelExperiment4
from src.experiments.experiment_3 import ModelExperiment3
from src.experiments.experiment_2 import ModelExperiment2
from src.experiments.experiment_1 import ModelExperiment1

base_experiment = ModelExperiment4(3)
BreakoutGame(base_experiment, True, 'human').play()
