import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def visualize_data(filename, height, width):
    df = pd.read_csv(f"data/{filename}", header=None)
    data = df.values
    samples, _ = data.shape
    print(samples)
    plt.figure(1, figsize=(height, samples * width))
    i = 0
    for im in data:
        i += 1
        plt.subplot(1, samples, i)
        plt.imshow(im.reshape(height, width), cmap='gray')


def visualize_result(original, noised, result, height, width, title=None):
    fig, axs = plt.subplots(1, 3, constrained_layout=True)
    if title:
        fig.suptitle(title, fontsize=16)

    axs[0].imshow(original.reshape(height, width), cmap='gray')
    axs[0].set_title('Original')
    axs[1].imshow(noised.reshape(height, width), cmap='gray')
    axs[1].set_title('Noised')
    axs[2].imshow(result.reshape(height, width), cmap='gray')
    axs[2].set_title('Result')


def visualize_result_step(prev_y, next_y, height, width, original, title):
    fig, axs = plt.subplots(1, 3, constrained_layout=True)
    fig.suptitle(title, fontsize=16)
    axs[0].imshow(original.reshape(height, width), cmap='gray')
    axs[0].set_title('Original')
    axs[1].imshow(prev_y.reshape(height, width), cmap='gray')
    axs[1].set_title('Previous')
    axs[2].imshow(next_y.reshape(height, width), cmap='gray')
    axs[2].set_title('Current')
    for ax_j in axs:
        ax_j.set_xticks([])
        ax_j.set_yticks([])
        ax_j.patch.set_edgecolor('black')
        ax_j.patch.set_linewidth('2')


def show(x, col):
    n = len(x)
    for i in range(0, n):
        if i % col == 0:
            print()
        if x[i] == 1:
            print('*', end=''),
        else:
            print(' ', end=''),
    print()


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


def get_test_data(x, points_count, random_generator):
    test_data = np.array(x)
    n = len(x)
    noise_position = list(range(n))
    random_generator.shuffle(noise_position)
    for k in noise_position[:points_count]:  # invert points_count points in the pattern
        test_data[k] = -test_data[k]
    test_data = test_data.reshape((n, 1))
    return test_data
