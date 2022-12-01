import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from data_partition import data_partition


def get_mean_by_channel(dataset):
    """Compute the mean value of dataset by channel."""
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)

    for inputs, targets in dataloader:
        input = inputs.reshape((3, 32, 32))
        print(input)
        for i in range(3):
            mean[i] += input[i, :, :].mean()
    mean.div_(len(dataset))
    return mean


# tensor(0.4247, 0.4149, 0.3838)
# tensor(0.4915, 0.4821, 0.4464)

transform_train = transforms.Compose([
    # transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(0.4914, 0.4822, 0.4465), inplace=False),
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(
    root='./dataset', train=True, download=True, transform=transform_train)

mean = get_mean_by_channel(trainset)
print(mean)




