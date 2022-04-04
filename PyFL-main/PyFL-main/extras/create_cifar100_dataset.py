import os
import sys
import cv2
import numpy as np
from math import floor
import torch
import torch.optim
import torch.utils.data
import torch.nn.parallel
import torchvision.datasets as datasets
import torchvision.transforms as transforms


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


def interpolate(data):
    data = data.numpy().astype(np.float32) * 255
    data = data.T
    bicubic_img = cv2.resize(data, None, fx=7, fy=7, interpolation=cv2.INTER_CUBIC)
    # cv2_imshow(bicubic_img)
    return bicubic_img.T / 255


def create_cifar100_partitions(nr_of_datasets):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='data/clients', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            # normalize,
        ]), download=True),
        batch_size=50000, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='data/clients', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            # normalize,
        ])),
        batch_size=10000, shuffle=True, pin_memory=True)
    dataset = {}
    for x, y in train_loader:
        new_data = np.zeros((50000, 3, 224, 224)).astype(np.float32)
        for index, data in enumerate(x):
            new_data[index, :, :, :] = interpolate(data)
        dataset["x_train"], dataset["y_train"] = new_data, y.numpy()
    for x, y in val_loader:
        new_data = np.zeros((10000, 3, 224, 224)).astype(np.float32)
        for index, data in enumerate(x):
            new_data[index, :, :, :] = interpolate(data)
        dataset["x_test"], dataset["y_test"] = new_data, y.numpy()
    # np.savez_compressed('data/clients/cifar100.npz',dataset,allow_pickle=False, pickle_kwargs={"protocol":4})
    print("CREATING {} PARTITIONS INSIDE {}/data/clients".format(nr_of_datasets, os.getcwd()))
    for key, val in dataset.items():
        dataset[key] = splitset(key, val, nr_of_datasets)
    if not os.path.exists(os.getcwd() + '/data/clients'):
        os.mkdir(os.getcwd() + '/data/clients')
    for i in range(nr_of_datasets):
        if not os.path.exists('data/clients/{}'.format(str(i + 1))):
            os.mkdir('data/clients/{}'.format(str(i + 1)))

        np.savez('data/clients/{}'.format(str(i + 1)) + '/data.npz',
                 x_train=dataset['x_train'][i],
                 y_train=dataset['y_train'][i],
                 x_test=dataset['x_test'][i],
                 y_test=dataset['y_test'][i])
    print("DONE")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        nr_of_datasets = 10
    else:
        nr_of_datasets = int(sys.argv[1])
    create_cifar100_partitions(nr_of_datasets)
