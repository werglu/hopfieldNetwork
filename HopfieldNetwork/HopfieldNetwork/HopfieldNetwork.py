import numpy as np
import pandas as pd
import random
from utilities import read_data, visualize_result, show
import matplotlib.pyplot as plt


def get_weights_hebb(x):
    w = np.zeros([len(x), len(x)])
    for i in range(len(x)):
        for j in range(i, len(x)):
            if i == j:
                w[i, j] = 0
            else:
                w[i, j] = x[i] * x[j]
                w[j, i] = w[i, j]
    return w


def get_weights_matrix_hebb_rule(X, M, N):
    T = np.dot(X, np.transpose(X)) - M * np.eye(N)
    T = np.divide(T, N)
    return T


def get_max_iter_count(M):
    return 4**(2**(M-1))


def get_test_data(x, points_count, random_generator):
    test_data = np.array(x)
    n = len(x)
    noise_position = list(range(n))
    random_generator.shuffle(noise_position)
    for k in noise_position[:points_count]:  # invert points_count points in the pattern
        test_data[k] = -test_data[k]
    test_data = test_data.reshape((n, 1))
    return test_data

# TODO - dlaczego theta jest 0.5? mi to psuje wyniki
def process(w, y_vec, theta=0.5, time=100):
    for s in range(time):
        m = len(y_vec)
        # i = random.randint(0,m-1)
        for i in range(m):
            u = np.dot(w[i][:],
                       y_vec) - theta  # COMMENT: w sumie nie wiem czy to jest sync czy async ale uaktualniam wszytskie po prostu
            if u >= 0:
                y_vec[i] = 1
            elif u < 0:
                y_vec[i] = -1

    return y_vec


def recognition_phase_synchronous(weights, y_vec, max_iter_count):
    y_prev = np.copy(y_vec)
    y_prev[0] -= 1
    curr_iter = 1
    while (not np.array_equal(y_prev, y_vec)) and curr_iter < max_iter_count:
        print(curr_iter)
        y_prev = np.copy(y_vec)
        u = np.dot(weights, y_vec)
        for i in range(len(u)):
            if u[i] >= 0:
                y_vec[i] = 1
            else:
                y_vec[i] = -1
        curr_iter += 1
    if np.array_equal(y_prev, y_vec):
        print("Model convergence")
    else:
        print("Iter exceeded")
    return y_vec

# df_test = read_data("animals-14x9.csv")
# col = 9
# data = np.array(df_test);
# neurons_count = len(data[1]);
# random_generator = np.random.default_rng(seed=123)
#
# for i in range(1, len(data)):
#     weights = weights + get_weights_hebb(data[i])
#
# weights = weights / len(data)  # divide when hebb rule
# print(weights)

# for i in range(1, len(data)):
#     print(i)
#     show(data[i])
#     data_test = get_test_data(x[i], 15)
#     print('Test data:')
#     show(data_test)
#     print('After process:')
#     show(process(weights, data_test))
#     print()

# TEST CASES
filename = "animals-14x9.csv"
height, width = 14, 9
# filename = "large-25x25.csv"
# height, width = 25, 25
# filename = "small-7x7.csv"
# height, width = 7, 7

df_test = read_data(filename)
data = df_test.values
sample_count_M, neurons_count_N = data.shape
data = data.T
# print(data.shape)
T = get_weights_matrix_hebb_rule(data, M=sample_count_M, N=neurons_count_N)
# print(T.shape)

random_generator = np.random.default_rng(seed=123)
sample_id = 1
noise_percentage = 0.1
noise_changes_count = int(noise_percentage * neurons_count_N)

sample = np.reshape(data[:, sample_id], (neurons_count_N, 1))
sample_test = get_test_data(np.copy(sample), noise_changes_count, random_generator)
max_iter_count = get_max_iter_count(sample_count_M)

result_synchronous = recognition_phase_synchronous(T, np.copy(sample_test), max_iter_count)
print("Accuracy synchronous: {0}".format(np.sum(sample == result_synchronous) / neurons_count_N))
visualize_result(sample, sample_test, result_synchronous, height, width)
plt.show()


result_asynchronous = process(T, np.copy(sample_test))
print("Accuracy asynchronous: {0}".format(np.sum(sample == result_asynchronous) / neurons_count_N))
visualize_result(sample, sample_test, result_asynchronous, height, width)
plt.show()
