import os
import numpy as np


if not os.path.exists('../data/clients'):
    os.mkdir('../data/clients')
nr_of_datasets = 10
per = 0.8
for i in range(nr_of_datasets):
    print(i+1)
    package = np.load("/home/diptanshu18232/Imagenet/train_data_batch_" + str(i + 1) + ".npz")
    if not os.path.exists('data/clients/{}'.format(str(i + 1))):
        os.mkdir('data/clients/{}'.format(str(i + 1)))
    np.savez('data/clients/{}'.format(str(i + 1)) + '/imagenet.npz',
             x_train=package['data'][:int(per * package["data"].shape[0]), :],
             y_train=package['labels'][:int(per * package["data"].shape[0])],
             x_test=package['data'][int(per * package["data"].shape[0]):, :],
             y_test=package['labels'][int(per * package["data"].shape[0]):])
print("DONE")
