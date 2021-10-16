import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys
import numpy as np
import cpp as cpp

seq_length = 28
input_size = 28
hidden_size = 128
num_layers = 1
num_classes = 10
batch_size = 1
num_epochs = 2
learning_rate = 0.01

data_path = "/media/lee/ESD-ISO/script_test/Data/mnist/"
train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root=data_path, train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = cpp.rnn(0.01, 28, 28, 128, 10)

total_step = len(train_loader)
iter_loss = 0
interval = 1
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(seq_length, batch_size, input_size).detach().numpy()
        labels = labels.detach().numpy()
        labels = cpp.NdArray(labels)
        
        hprev = np.zeros((hidden_size, 1))
        outputs = model.forward(images, hprev)
        # Y, loss = model.cross_entropy_loss(outputs, labels)
        print("{}".format(i))

        # gradients = model.backward(model.deriv_softmax(Y, labels))
        # model.optimizer(gradients)
        # iter_loss += np.sum(loss.getNumpyArray())
        # if (i + 1) % interval == 0:
        #     print("epoch {}/{} iter {}/{} loss {:.4f}".format(epoch + 1, num_epochs, i + 1, total_step, iter_loss / interval))
        #     iter_loss = 0
        break
    break