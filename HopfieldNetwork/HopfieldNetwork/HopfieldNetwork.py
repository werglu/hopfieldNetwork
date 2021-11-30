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

    def _get_weights_ij_hebb_rule(self):
        x = self.X
        n = self.neurons_count
        t = np.zeros((n, n))
        for i in range(0, n):
            for j in range(0, n):
                t[i, j] = np.matmul(x[i], x[j]) / n
        np.fill_diagonal(t, 0)
        return t

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
                if not np.isfinite(t).all():
                    t = t_prev
                    return t
        return t

    def get_weights(self, nu=None, iter_count=None, eps=1e-14):
        if self.rule == LearningRule.Hebb:
            if self.neurons_count < 10000:
                return self._get_weights_matrix_hebb_rule()
            else:
                return self._get_weights_ij_hebb_rule()
        else:
            return self._get_weights_matrix_oja_rule(nu, iter_count, eps)

    def recognize(self, weights, y_vec, max_iter_count=None, show_data=None, height=None, weight=None, original=None):

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
                for i in range(len(u)):
                    y_vec[i] = self._update_neuron_value(u[i])
            else:
                permutation = list(range(0, self.neurons_count))
                self.random_generator.shuffle(permutation)
                for neu_i in permutation:
                    y_vec[neu_i] = self._update_neuron_value(u[neu_i])
                    u = np.matmul(weights, y_vec)

            if show_data:
                visualize_result_step(y_prev, y_vec, height, weight, original, "Iteration {0}".format(curr_iter))
                plt.show()

            curr_iter += 1

            convergence = np.array_equal(y_prev, y_vec)
        if convergence:
            print("Model convergence at {0} iter".format(curr_iter))
        else:
            print("Iter exceeded")
        return y_vec

    def recognize_and_animate(self, weights, y_vec, height, weight, original, max_iter_count=None):
        self.recognize(weights=weights, y_vec=y_vec, max_iter_count=max_iter_count,
                       height=height, weight=weight, original=original, show_data=True)

    def _update_neuron_value(self, val):
        if val >= 0:
            return 1
        else:
            return -1

    def set_mode(self, new_mode):
        self.mode = new_mode

    def set_learning_rule(self, rule):
        self.rule = rule
