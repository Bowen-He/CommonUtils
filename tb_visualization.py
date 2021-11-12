import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

dir_1 = "./training_results_For_Dopamine/DQN/Breakout/"
dir_2 = "./training_results_For_Dopamine/C51/Breakout/"


def get_statistics(array):
    median = np.median(array, axis=0)
    means = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    N = array.shape[0]
    top = means + (std / np.sqrt(N))
    bot = means - (std / np.sqrt(N))
    return median, means, top, bot


def get_data(path: str) -> list:
    summaries = tf.compat.v1.train.summary_iterator(path)
    values = []
    for e in summaries:
        for v in e.summary.value:
            if v.tag == 'Eval/AverageReturns':
                values.append(v.simple_value)
    return values


def main():
    value_DQN = []
    mini_len = 999
    for item in os.listdir(dir_1):
        path = dir_1 + item
        temp_list = np.convolve(get_data(path), np.ones(10)/10, mode='valid')
        mini_len = temp_list.shape[0] if temp_list.shape[0] < mini_len else mini_len
        value_DQN.append(temp_list)
    for i, item in enumerate(value_DQN):
        value_DQN[i] = item[0:mini_len]

    value_C51 = []
    mini_len = 999
    for item in os.listdir(dir_2):
        path = dir_2 + item
        temp_list = np.convolve(get_data(path), np.ones(10)/10, mode='valid')
        mini_len = temp_list.shape[0] if temp_list.shape[0] < mini_len else mini_len
        value_C51.append(temp_list)
    for i, item in enumerate(value_C51):
        value_C51[i] = item[0:mini_len]

    median, mean, top, bottom = get_statistics(np.array(value_DQN))
    plt.plot(mean, linewidth=2, label="DQN", alpha=0.9, color="blue")
    plt.fill_between(range(mean.shape[0]), top, bottom, alpha=0.2, color="blue")

    median, mean, top, bottom = get_statistics(np.array(value_C51))
    plt.plot(mean, linewidth=2, label="C51", alpha=0.9, color="red")
    plt.fill_between(range(mean.shape[0]), top, bottom, alpha=0.2, color="red")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

