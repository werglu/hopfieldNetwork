from HopfieldNetwork import HopfieldNetwork, LearningRule, Mode
from data_manager import *


random_seed = 123

x1 = [-1, -1]
x2 = [1, 1]

sample_count_m = 2
neurons_count_n = 2
height = 2
width = 1
data = np.column_stack((x1, x2))

network = HopfieldNetwork(LearningRule.Hebb, Mode.Synchronous, data, sample_count_m, neurons_count_n, random_seed)
network.set_learning_rule(LearningRule.Hebb)
T_Hebb = network.get_weights()

sample = np.array([1, -1]).reshape(2, 1)
sample_test = sample

network.set_mode(Mode.Synchronous)
result_synchronous = network.recognize_and_animate(T_Hebb, np.copy(sample_test),
                                                   height, width, sample, max_iter_count=10)

print("Accuracy synchronous: {0}".format(np.sum(sample == result_synchronous) / neurons_count_n))

