import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
from npLSTM import npLSTM

seq_length = 28
input_size = 28
hidden_size = 256
num_layers = 1
num_classes = 10
batch_size = 1
num_epochs = 1
learning_rate = 0.01
start_time = 0

data_path = "../../../data/mnist"
train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root=data_path, train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = npLSTM(learning_rate, seq_length, input_size, hidden_size, num_classes)

total_step = len(train_loader)
iter_loss = 0
interval = 1000
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(seq_length, batch_size, input_size).detach().numpy()
        labels = labels.detach().numpy()
        
        hprev = np.zeros((hidden_size, 1))
        cprev = np.zeros((hidden_size, 1))
        outputs = model.forward(images, hprev, cprev)
        Y, loss = model.cross_entropy_loss(outputs, labels)
        gradients = model.backward(model.deriv_softmax(Y, labels))
        model.optimizer_step(gradients)
        iter_loss += np.sum(loss)
        if(i + 1) % interval == 0:
            print("epoch {}/{} iter {}/{} loss {:.4f}".format(epoch + 1, num_epochs, i + 1, total_step, iter_loss / interval))
            iter_loss = 0

correct = 0
total = 0
for images, labels in test_loader:
    images = images.reshape(seq_length, batch_size, input_size).detach().numpy()
    labels = labels.detach().numpy()
    
    hprev = np.zeros((hidden_size, 1))
    cprev = np.zeros((hidden_size, 1))
    outputs = model.forward(images, hprev, cprev)
    pred = model.predict(outputs)
    total += labels.shape[0]
    correct += (pred == labels).sum().item()

print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 