import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from utilities import visualize_result_step


class Mode(Enum):
    Synchronous = 1
    Asynchronous = 2


class LearningRule(Enum):
    Hebb = 1
    Oja = 2


class HopfieldNetwork:

    def __init__(self, rule, mode, X, vectors_count, sample_size, random_seed):
        self.rule = rule
        self.mode = mode
        self.X = X
        self.vectors_count = vectors_count
        self.neurons_count = sample_size
        self.random_generator = np.random.default_rng(seed=random_seed)

    def _get_max_iter_count(self):
        m = self.vectors_count
        return 4**(2**(m-1))

    def _get_weights_matrix_hebb_rule(self):
        X = self.X
        m = self.vectors_count
        n = self.neurons_count

        T = np.matmul(X, np.transpose(X)) - m * np.eye(n)
        T = np.divide(T, n)
        return T

    def _get_weights_matrix_oja_rule(self, nu, iter_count, eps=1e-14):
        t = self._get_weights_matrix_hebb_rule()

        X = self.X
        m = self.vectors_count
        for it in range(iter_count):
            for i in range(m):
                x = X[:, i]
                t_prev = np.copy(t)
                y = np.matmul(x, t)
                t += nu * np.outer(y, (x - np.matmul(y, t)))
                if np.linalg.norm(t - t_prev) < eps:
                    break
        return t

    def get_weights(self, nu=None, iter_count=None, eps=1e-14):
        if self.rule == LearningRule.Hebb:
            return self._get_weights_matrix_hebb_rule()
        else:
            return self._get_weights_matrix_oja_rule(nu, iter_count, eps)

    def recognize(self, weights, y_vec, max_iter_count=None, show_data=None):

        if not max_iter_count:
            max_iter_count = self._get_max_iter_count()

        curr_iter = 1
        convergence = False
        while (not convergence) and curr_iter < max_iter_count:
            if show_data:
                print(curr_iter)
            y_prev = np.copy(y_vec)
            u = np.matmul(weights, y_vec)

            if self.mode == Mode.Synchronous:
                self._update_synchronous(y_vec, u)
            else:
                self._update_asynchronous(y_vec, u)

            curr_iter += 1

            convergence = np.array_equal(y_prev, y_vec, )
        if convergence:
            print("Model convergence at {0} iter".format(curr_iter))
        else:
            print("Iter exceeded")
        return y_vec

    def recognize_and_animate(self, weights, y_vec, height, weight, original, max_iter_count=None):

        if not max_iter_count:
            max_iter_count = self._get_max_iter_count()

        curr_iter = 1
        convergence = False
        while (not convergence) and curr_iter < max_iter_count:
            y_prev = np.copy(y_vec)
            u = np.matmul(weights, y_vec)

            if self.mode == Mode.Synchronous:
                self._update_synchronous(y_vec, u)
            else:
                self._update_asynchronous(y_vec, u)

            convergence = np.array_equal(y_prev, y_vec)

            visualize_result_step(y_prev, y_vec, height, weight, original, "Iteration {0}".format(curr_iter))
            plt.show()
            curr_iter += 1

        if convergence:
            print("Model convergence at {0} iter".format(curr_iter))
        else:
            print("Iter exceeded")
        return y_vec

    def _update_neuron_value(self, val):
        if val >= 0:
            return 1
        else:
            return -1

    def _update_synchronous(self, y, u):
        for i in range(len(u)):
            y[i] = self._update_neuron_value(u[i])

    def _update_asynchronous(self, y, u):
        neu_i = self.random_generator.integers(0, len(y))
        y[neu_i] = self._update_neuron_value(u[neu_i])

    def set_mode(self, new_mode):
        self.mode = new_mode

    def set_learning_rule(self, rule):
        self.rule = rule

    # def recognition_phase_synchronous(slef, weights, y_vec, max_iter_count):
    #     y_prev = np.copy(y_vec)
    #     y_prev[0] -= 1
    #     curr_iter = 1
    #     while (not np.array_equal(y_prev, y_vec)) and curr_iter < max_iter_count:
    #         print(curr_iter)
    #         y_prev = np.copy(y_vec)
    #         u = np.matmul(weights, y_vec)
    #         update_sychronous(y_vec, u)
    #
    #         curr_iter += 1
    #     if np.array_equal(y_prev, y_vec):
    #         print("Model convergence")
    #     else:
    #         print("Iter exceeded")
    #     return y_vec
    #
    #
    # def recognition_phase_asynchronous(weights, y_vec, max_iter_count, random_generator):
    #     y_prev = np.copy(y_vec)
    #     y_prev[0] -= 1
    #     curr_iter = 1
    #     while (not np.array_equal(y_prev, y_vec)) and curr_iter < max_iter_count:
    #         print(curr_iter)
    #         y_prev = np.copy(y_vec)
    #         u = np.matmul(weights, y_vec)
    #
    #         # Update one neuron
    #         update_asychronous(y_vec, u, random_generator)
    #
    #         curr_iter += 1
    #     if np.array_equal(y_prev, y_vec):
    #         print("Model convergence")
    #     else:
    #         print("Iter exceeded")
    #     return y_vec


# # TODO - dlaczego theta jest 0.5? mi to psuje wyniki
# def process(w, y_vec, theta=0.5, time=100):
#     for s in range(time):
#         m = len(y_vec)
#         # i = random.randint(0,m-1)
#         for i in range(m):
#             u = np.matmul(w[i][:],
#                        y_vec) - theta
# COMMENT: w sumie nie wiem czy to jest sync czy async ale uaktualniam wszytskie po prostu
#             if u >= 0:
#                 y_vec[i] = 1
#             elif u < 0:
#                 y_vec[i] = -1
#     return y_vec
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


