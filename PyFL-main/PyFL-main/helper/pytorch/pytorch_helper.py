# import os
# import tempfile
# from functools import reduce
import os
import pickle
from collections import OrderedDict
import numpy as np
import torch
from PIL import Image
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import VisionDataset
from typing import Any, Callable, Optional, Tuple


class PytorchHelper:

    def increment_average(self, model, model_next, n):
        """ Update an incremental average. """
        w = OrderedDict()
        for name in model.keys():
            tensorDiff = model_next[name] - model[name]
            w[name] = model[name] + tensorDiff / n
        return w

    # def get_tmp_path(self):
    #     fd, path = tempfile.mkstemp(suffix='.npz')
    #     os.close(fd)
    #     return path

    def get_tensor_diff(self, model, base_model):
        w = OrderedDict()
        for name in model.keys():
            w[name] = model[name] - base_model[name]
        return w

    def add_base_model(self, tensordiff, base_model, learning_rate):
        w = OrderedDict()
        for name in tensordiff.keys():
            w[name] = learning_rate * tensordiff[name] + base_model[name]
        return w

    def save_model(self, weights_dict, path=None):
        if not path:
            path = self.get_tmp_path()
        np.savez_compressed(path, **weights_dict)
        return path

    def load_model(self, path="weights.npz"):
        b = np.load(path)
        weights_np = OrderedDict()
        for i in b.files:
            weights_np[i] = b[i]
        return weights_np

    def read_data(self, dataset, data_path, trainset):
        if dataset == "Imagenet":
            return self.read_data_imagenet(data_path, trainset)
        elif dataset == "mnist":
            return self.read_data_mnist(data_path, trainset)
        elif dataset == "cifar10":
            return self.temp_read_data_cifar10(trainset)
        elif dataset == "cifar100":
            return self.read_data_cifar100(data_path, trainset)

    def read_data_cifar100(self, data_path, trainset=True):
        pack = np.load(data_path)
        if trainset:
            X = pack['x_train']
            y = pack['y_train']
        else:
            X = pack['x_test']
            y = pack['y_test']
        X = X.astype('float32')
        y = y.astype('int64')
        transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        tensor_x = torch.Tensor(X)
        tensor_y = torch.from_numpy(y)
        tensor_x = transform(tensor_x)
        dataset = TensorDataset(tensor_x, tensor_y)
        return dataset

    def temp_read_data_cifar10(self, trainset=True, test_sample=3000, train_sample=10000):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if trainset:
            dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True)
            sample = np.random.permutation(len(dataset))[:train_sample]
        else:
            dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
            sample = np.random.permutation(len(dataset))[:test_sample]
        return torch.utils.data.Subset(dataset, sample)

    def read_data_cifar10(self, data_path, trainset=True):
        pack = torch.load(data_path)
        if trainset:
            X = pack['x_train']
            print(X.dtype)
            y = pack['y_train']
        else:
            X = pack['x_test']
            y = pack['y_test']
        # X = X.astype('float32')
        # y = y.astype('int64')
        # tensor_x = torch.Tensor(X)
        # tensor_y = torch.from_numpy(y)
        dataset = TensorDataset(X, y)
        return dataset

    def read_data_mnist(self, data_path, trainset=True):
        pack = np.load(data_path)
        if trainset:
            X = pack['x_train']
            y = pack['y_train']
        else:
            X = pack['x_test']
            y = pack['y_test']
        X = X.astype('float32')
        y = y.astype('int64')
        X = np.expand_dims(X, 1)
        X /= 255
        tensor_x = torch.Tensor(X)
        tensor_y = torch.from_numpy(y)
        dataset = TensorDataset(tensor_x, tensor_y)
        return dataset

    # def read_data_mnist(self, data_path, trainset=True):
    #     pack = np.load(data_path)
    #     if trainset:
    #         X = pack['x_train']
    #         y = pack['y_train']
    #     else:
    #         X = pack['x_test']
    #         y = pack['y_test']
    #     X = X.astype('float32')
    #     y = y.astype('int64')
    #     print(X.shape)
    #     X = np.repeat(X , 3 , axis=2).reshape(len(X) , 3,28,28)
    #     X /= 255
    #     # print(X.shape , y.shape)
    #     tensor_x = torch.Tensor(X)
    #     tensor_y = torch.from_numpy(y)
    #     dataset = TensorDataset(tensor_x, tensor_y)
    #     return dataset

    def read_data_imagenet(self, data_path, trainset=True):
        pack = np.load(data_path)
        if trainset:
            X = pack['x_train']
            y = pack['y_train']
        else:
            X = pack['x_test']
            y = pack['y_test']
        X = X.astype('float32')
        y = y.astype('int64') - 1
        X = X.reshape(X.shape[0], 3, 64, 64)
        X /= 255
        tensor_x = torch.Tensor(X)
        tensor_y = torch.from_numpy(y)
        dataset = TensorDataset(tensor_x, tensor_y)
        return dataset

    # def load_model_from_BytesIO(self, model_bytesio):
    #     """ Load a model from a BytesIO object. """
    #     path = self.get_tmp_path()
    #     with open(path, 'wb') as fh:
    #         fh.write(model_bytesio)
    #         fh.flush()
    #     model = self.load_model(path)
    #     os.unlink(path)
    #     return model
    #
    # def serialize_model_to_BytesIO(self, model):
    #     outfile_name = self.save_model(model)
    #
    #     from io import BytesIO
    #     a = BytesIO()
    #     a.seek(0, 0)
    #     with open(outfile_name, 'rb') as f:
    #         a.write(f.read())
    #     os.unlink(outfile_name)
    #     return a


class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        return True

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")
