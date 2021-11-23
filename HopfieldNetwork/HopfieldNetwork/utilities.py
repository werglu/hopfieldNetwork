import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


def read_data(filename):
    df = pd.read_csv(f"data/{filename}", header=None)
    return df


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


def visualize_result(original, noised, result, height, width):
    plt.figure(1, figsize=(height, 3 * width))
    plt.subplot(1, 3, 1)
    plt.imshow(original.reshape(height, width), cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(noised.reshape(height, width), cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(result.reshape(height, width), cmap='gray')


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