import numpy as np
from utilities import visualize_result, show, get_test_data
import matplotlib.pyplot as plt
from HopfieldNetwork import HopfieldNetwork, LearningRule, Mode
from data_manager import get_set_small_7x7

# TEST CASE
data, sample_count_m, neurons_count_n, height, width = get_set_small_7x7()

random_seed = 1234

network = HopfieldNetwork(LearningRule.Hebb, Mode.Synchronous, data, sample_count_m, neurons_count_n, random_seed)
T_Hebb = network.get_weights()

network.set_learning_rule(LearningRule.Oja)
T_Oja = network.get_weights(nu=0.001, iter_count=100, eps=1e-14)

print("Weights done")
random_generator = np.random.default_rng(seed=123)

for i in range(0, sample_count_m):
    sample_id = i
    noise_percentage = 0.1
    noise_changes_count = int(noise_percentage * neurons_count_n)

    sample = np.reshape(data[:, sample_id], (neurons_count_n, 1))
    sample_test = get_test_data(np.copy(sample), noise_changes_count, random_generator)

    network.set_mode(Mode.Synchronous)
    result_synchronous = network.recognize(T_Hebb, np.copy(sample_test))
    print("Accuracy synchronous: {0}".format(np.sum(sample == result_synchronous) / neurons_count_n))
    visualize_result(sample, sample_test, result_synchronous, height, width)
    plt.show()

# network.set_mode(Mode.Asynchronous)
# times = 1000
# learning_rate = 0.0001
# print(learning_rate)
# result_asynchronous = network.recognize(T_Oja, np.copy(sample_test))
# print("Accuracy asynchronous: {0}".format(np.sum(sample == result_asynchronous) / neurons_count_N))
# print("Accuracy asynchronous with test: {0}".format(np.sum(sample_test == result_asynchronous) / neurons_count_N))
# visualize_result(sample, sample_test, result_asynchronous, height, width)
# plt.show()
