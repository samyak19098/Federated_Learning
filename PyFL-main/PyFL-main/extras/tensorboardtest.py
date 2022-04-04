from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from torch.utils.data import TensorDataset
import torchvision
import matplotlib.pyplot as plt

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('temp')

pack = np.load("short.npz")
print(pack.files)
X = pack['data']
y = pack['labels']
X = X.astype('float32')
print(X.shape)
y = y.astype('int64')
X = X.reshape(X.shape[0], 3, 64, 64)
X /= 255
tensor_x = torch.Tensor(X)
print(tensor_x.size())
tensor_y = torch.from_numpy(y)
dataset = TensorDataset(tensor_x, tensor_y)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=40,
                                          shuffle=True, num_workers=2)


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


dataiter = iter(trainloader)
images, labels = dataiter.next()

# create grid of images
img_grid = torchvision.utils.make_grid(images)
print(img_grid.shape)
# show images
matplotlib_imshow(img_grid, one_channel=False)

# write to tensorboard
writer.add_image('ImageNet Train Images', img_grid)
writer.close()
