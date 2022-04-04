import os
import sys
import numpy as np
from math import floor


def splitset(key, dataset, parts):
    """Partition data into "parts" partitions"""
    n = dataset.shape[0]
    local_n = floor(n / parts)
    print(key)
    print("Original shape : ", dataset.shape)
    arr = np.array(np.split(dataset, parts))
    print("Splitted shape : ", arr.shape)
    result = []
    for i in range(parts):
        result.append(dataset[i * local_n: (i + 1) * local_n])
    return np.array(result)


def create_mnist_partitions(nr_of_datasets):
    package = np.load(os.getcwd() + "/data/mnist.npz")
    data = {}
    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
    for key, val in package.items():
        data[key] = val
    data["x_test"], data["y_test"] = unison_shuffled_copies(data["x_test"], data["y_test"])
    data["x_train"], data["y_train"] = unison_shuffled_copies(data["x_train"], data["y_train"])
    print("CREATING {} PARTITIONS INSIDE {}/data/clients".format(nr_of_datasets, os.getcwd()))
    for key, val in data.items():
        data[key] = splitset(key, val, nr_of_datasets)
    if not os.path.exists(os.getcwd() + '/data/clients'):
        os.mkdir(os.getcwd() + '/data/clients')
    for i in range(nr_of_datasets):
        if not os.path.exists('data/clients/{}'.format(str(i + 1))):
            os.mkdir('data/clients/{}'.format(str(i + 1)))
        np.savez('data/clients/{}'.format(str(i + 1)) + '/data.npz',
                 x_train=data['x_train'][i],
                 y_train=data['y_train'][i],
                 x_test=data['x_test'][i],
                 y_test=data['y_test'][i])
    print("DONE")


if __name__ == '__main__':

    if len(sys.argv) < 2:
        nr_of_datasets = 10
    else:
        nr_of_datasets = int(sys.argv[1])
    create_mnist_partitions(nr_of_datasets)
