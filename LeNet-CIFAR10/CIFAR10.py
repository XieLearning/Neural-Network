# _*_coding:utf-8_*_
# Author   :xie
# Time     :18-11-17 
# File     :CIFAR10.py
# IDE      :PyCharm

import torch as t
from torch import optim
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

Max_epoch = 10


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def preprocessing():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = tv.datasets.CIFAR10(root='/home/xie/文档/datasets/', train=True, download=True, transform=transform)
    train_data = t.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
    test_set = tv.datasets.CIFAR10(root='/home/xie/文档/datasets/', train=False, download=True, transform=transform)
    test_data = t.utils.data.DataLoader(test_set, batch_size=16, shuffle=False)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_data, test_data, classes


def train(train_data):
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(Max_epoch):
        running_loss = 0

        for i, data in enumerate(train_data, 0):
            # input data
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            # make the gradient to zero
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # update the parameters
            optimizer.step()

            # print the log information
            running_loss += loss.item()
            if i % 125 == 124:
                print('[%d, %5d] loss:%.3f' % (epoch+1, i+1, running_loss/125))
                running_loss = 0.0
    print('Finish Training!')
    return net


def test(net, test_data):
    correct, total = 0.0, 0

    for data in test_data:
        images, labels = data
        outputs = net(Variable(images))
        _, predicted = t.max(outputs.data, 1)  # get the largest data of each row
        total += labels.size(0)
        correct += (predicted == labels).sum()

    return float(correct)/total


if __name__ == '__main__':
    train_data, test_data, classes = preprocessing()
    net = train(train_data)
    test_accuracy = test(net, test_data)
    print('The accuracy of %d pictures is: %.4f' % (len(test_data)*4, test_accuracy))



