import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

from ResNet import ResNet18
from data_partition import data_partition


def train(net, criterion, trainloader, scheduler=None):
    device = 'cpu'
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 50 == 0:
            print("iteration : %3d, loss : %0.4f, accuracy : %2.2f" % (
                batch_idx + 1, train_loss / (batch_idx + 1), 100. * correct / total))

    if scheduler is not None:
        scheduler.step()

    return train_loss / (batch_idx + 1), 100. * correct / total


def test(net, criterion, testloader):
    device = 'cpu'
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return test_loss / (batch_idx + 1), 100. * correct / total


# data import
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),

    # value=(0.4914, 0.4822, 0.4465)
    transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(0.4914, 0.4822, 0.4465), inplace=False),

    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./dataset', train=True, download=True, transform=transform_train)

testset = torchvision.datasets.CIFAR10(
    root='./dataset', train=False, download=True, transform=transform_test)

training_set, validation_set = data_partition(trainset)
print(len(training_set))
print(len(validation_set))
print(len(testset))

trainloader = torch.utils.data.DataLoader(training_set, batch_size=128, shuffle=True, num_workers=2)
validateloader = torch.utils.data.DataLoader(validation_set, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

# hyperparameter set
device = 'cpu'
config = {
    'epoch': 10,
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4
}

# train and validate
net = ResNet18().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=config['lr'],
                      momentum=config['momentum'], weight_decay=config['weight_decay'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

train_loss_list, train_acc_list, validate_loss_list, validate_acc_list = [], [], [], []

for epoch in range(1, config['epoch']):
    print('\nEpoch: %d' % epoch)
    train_loss, train_acc = train(net, criterion, trainloader)
    validate_loss, validate_acc = test(net, criterion, validateloader)

    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    validate_loss_list.append(validate_loss)
    validate_acc_list.append(validate_acc)

    print(("Epoch : %3d, training loss : %0.4f, training accuracy : %2.2f, test loss " +
           ": %0.4f, test accuracy : %2.2f") % (epoch, train_loss, train_acc, validate_loss, validate_acc))

test_loss, test_acc = test(net, criterion, testloader)
print("Test :  testing loss: %0.4f, testing accuracy: %2.2f" % (test_loss, test_acc))

plt.subplot(1, 2, 1)
plt.plot(range(len(train_loss_list)), train_loss_list, 'b')
plt.plot(range(len(validate_loss_list)), validate_loss_list, 'r')
plt.xlabel("Number of epochs")
plt.ylabel("Loss")
plt.title("ResNet: Loss vs Number of epochs")
plt.legend(['train', 'validate'])

plt.subplot(1, 2, 2)
plt.plot(range(len(train_acc_list)), train_acc_list, 'b')
plt.plot(range(len(validate_acc_list)), validate_acc_list, 'r')
plt.ylabel("Accuracy")
plt.title("ResNet: Accuracy vs Number of epochs")
plt.legend(['train', 'validate'])

plt.show()
