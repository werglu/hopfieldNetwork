from utilities import visualize_result, get_test_data
import matplotlib.pyplot as plt
from HopfieldNetwork import HopfieldNetwork, LearningRule, Mode
from data_manager import *


random_seed = 123

data, sample_count_m, neurons_count_n, height, width = get_set_large_25x25_plus()
network = HopfieldNetwork(LearningRule.Oja, Mode.Synchronous, data, sample_count_m, neurons_count_n, random_seed)
network.set_learning_rule(LearningRule.Oja)
T_Oja = network.get_weights(nu=0.001, iter_count=100, eps=1e-14)

random_generator = np.random.default_rng(seed=random_seed)
sample_id = 1
noise_percentage = 0.1
noise_changes_count = int(noise_percentage * neurons_count_n)

sample = np.reshape(data[:, sample_id], (neurons_count_n, 1))
sample_test = get_test_data(np.copy(sample), noise_changes_count, random_generator)

network.set_mode(Mode.Synchronous)
result_synchronous = network.recognize_and_animate(T_Oja, np.copy(sample_test),
                                                   height, width, sample, max_iter_count=1000)

print("Accuracy synchronous: {0}".format(np.sum(sample == result_synchronous) / neurons_count_n))

