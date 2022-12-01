import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data


def get_mean_by_channel(dataset):
    """Compute the mean value of dataset by channel."""
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)

    for inputs, targets in dataloader:
        input = inputs.reshape((3, 32, 32))
        for i in range(3):
            mean[i] += input[i, :, :].mean()
    mean.div_(len(dataset))
    return mean


def unpickle(file):
    with open("./dataset/cifar-10-batches-py/" + file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


dict = unpickle("data_batch_1")
data = dict[b'data']
labels = dict[b'labels']
# print(data)
# print(len(data))
# print(data[0].shape)

data = np.array(data)
images = data.reshape((10000, 3, 32, 32))

for i in range(25):
    for j in range(4):
        r = images[i * 4 + j][0].reshape(1024, 1)
        g = images[i * 4 + j][1].reshape(1024, 1)
        b = images[i * 4 + j][2].reshape(1024, 1)

        image = np.hstack((r, g, b))
        pic = image.reshape((32, 32, 3))

        plt.subplot(2, 2, j + 1)
        plt.imshow(pic)
    plt.show()

# index = 11
# r = images[index][0].reshape(1024, 1)
# g = images[index][1].reshape(1024, 1)
# b = images[index][2].reshape(1024, 1)
#
# image = np.hstack((r, g, b))
# pic = image.reshape((32, 32, 3))

# plt.imshow(pic)
# plt.savefig('./data characteristics/multiple-object-image/pic_1_12')
plt.show()
