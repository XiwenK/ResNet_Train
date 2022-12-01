import random

import torchvision
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class MyDataSet(Dataset):
    def __init__(self, data_list):
        self.sample_list = data_list

    def __getitem__(self, index) -> T_co:
        return self.sample_list[index]

    def __len__(self):
        return len(self.sample_list)


def data_partition(dataset):
    random.seed(0)
    num = dataset.__len__()
    index_set = random.sample([i for i in range(num)], round(num * 0.8))

    train_set, validate_set = [], []
    for i in range(num):
        item = dataset.__getitem__(i)
        if i in index_set:
            train_set.append(item)
        else:
            validate_set.append(item)

    training_set = MyDataSet(train_set)
    validation_set = MyDataSet(validate_set)

    return training_set, validation_set


def class_calculation(dataset):
    classes = ['plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    class_nums = [0 for _ in range(10)]

    for index, (data, labels) in enumerate(dataset):
        class_nums[labels] += 1

    for i in range(10):
        print("%s : %2.2f" % (classes[i], 100.0 * class_nums[i] / dataset.__len__()))


trainset = torchvision.datasets.CIFAR10(
    root='./dataset', train=True, download=True)

training_set, validation_set = data_partition(trainset)
class_calculation(training_set)
