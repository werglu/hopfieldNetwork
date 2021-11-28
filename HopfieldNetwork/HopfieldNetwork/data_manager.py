import pandas as pd
import numpy as np


def read_data(filename):
    df = pd.read_csv(f"data/{filename}", header=None)
    return df


def get_data_from_file(filename):
    df_test = read_data(filename)
    data = df_test.values
    sample_count_m, neurons_count_n = data.shape
    data = data.T
    return data, sample_count_m, neurons_count_n


def get_set_small_7x7():
    filename = "small-7x7.csv"
    height, width = 7, 7
    data, sample_count_m, neurons_count_n = get_data_from_file(filename)
    return data, sample_count_m, neurons_count_n, height, width


def get_set_animals_14x9():
    filename = "animals-14x9.csv"
    height, width = 14, 9
    data, sample_count_m, neurons_count_n = get_data_from_file(filename)
    return data, sample_count_m, neurons_count_n, height, width


def get_set_large_25x25():
    filename = "large-25x25.csv"
    height, width = 25, 25
    data, sample_count_m, neurons_count_n = get_data_from_file(filename)
    return data, sample_count_m, neurons_count_n, height, width


def get_set_large_25x25_plus():
    filename = "large-25x25.plus.csv"
    height, width = 25, 25
    data, sample_count_m, neurons_count_n = get_data_from_file(filename)
    return data, sample_count_m, neurons_count_n, height, width


def get_set_large_25x50():
    filename = "large-25x50.csv"
    height, width = 25, 50
    data, sample_count_m, neurons_count_n = get_data_from_file(filename)
    return data, sample_count_m, neurons_count_n, height, width


def get_set_letters_14x20():
    filename = "letters-14x20.csv"
    height, width = 14, 20
    data, sample_count_m, neurons_count_n = get_data_from_file(filename)
    return data, sample_count_m, neurons_count_n, height, width


def get_set_letters_abc_8x12():
    filename = "letters-abc-8x12.csv"
    height, width = 8, 12
    data, sample_count_m, neurons_count_n = get_data_from_file(filename)
    return data, sample_count_m, neurons_count_n, height, width


def get_set_OCRA_12x30():
    filename = "OCRA-12x30-cut.csv"
    height, width = 12, 30
    data, sample_count_m, neurons_count_n = get_data_from_file(filename)
    return data, sample_count_m, neurons_count_n, height, width


def delete_from_set(data, columns):
    new_m = data.shape[1] - len(columns)
    x = np.copy(data)
    for col in columns:
        x = np.delete(x, col, 1)
    return x, new_m
