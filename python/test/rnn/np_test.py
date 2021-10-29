import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
from npRnn import npRnn

seq_length = 28
input_size = 28
hidden_size = 256
num_layers = 1
num_classes = 10
batch_size = 1
num_epochs = 2
learning_rate = 0.01

data_path = "../../../data/mnist"
train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root=data_path, train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

def xavier_init(c1, c2, w=1, h=1, fc=False):
    fan_1 = c2 * w * h
    fan_2 = c1 * w * h
    ratio = np.sqrt(6.0 / (fan_1 + fan_2))
    params = ratio * (2 * np.random.random((c1, c2, w, h)) - 1)
    if fc:
        params = params.reshape(c1, c2)
    return params
    
total_step = len(train_loader)
interval = 1000

np_iter_loss = 0

np_U = xavier_init(hidden_size, input_size, fc=True)
np_W = xavier_init(hidden_size, hidden_size, fc=True)
np_V = xavier_init(hidden_size, hidden_size, fc=True)
np_FC_W = xavier_init(num_classes, hidden_size, fc=True)

print("start train numpy hidden_size {}".format(hidden_size))

np_model = npRnn(learning_rate, seq_length, hidden_size, num_classes, np_U, np_W, np_V, np_FC_W)
start_time = time.time()
for epoch in range(num_epochs):
    for i, (train_images, train_labels) in enumerate(train_loader):
        np_images = train_images.reshape(seq_length, batch_size, input_size).detach().numpy()
        np_labels = train_labels.detach().numpy()
        np_hprev = np.zeros((hidden_size, 1))

        np_outputs = np_model.forward(np_images, np_hprev)
        np_Y, np_loss = np_model.cross_entropy_loss(np_outputs, np_labels)
        np_dY = np_model.deriv_softmax(np_Y, np_labels)
        np_gradients = np_model.backward(np_dY)
        np_model.optimizer_step(np_gradients)
        np_iter_loss += np.sum(np_loss)

        if (i + 1) % interval == 0:
            print("numpy epoch {}/{} iter {}/{} loss {:.4f}".format(epoch + 1, num_epochs, i + 1, total_step, np_iter_loss / interval))
            print("elased time {}".format(time.time() - start_time))
            start_time = time.time()
            np_iter_loss = 0

np_correct = 0
np_total = 0

def softmax(x):
    e = np.exp(x)
    return e / np.sum(e)

def predict(outputs):
    return np.argmax(softmax(outputs), 0)

for test_images, test_labels in test_loader:
    np_images = test_images.reshape(seq_length, batch_size, input_size).detach().numpy()
    np_hprev = np.zeros((hidden_size, 1))
    labels = test_labels.detach().numpy()

    np_outputs = np_model.forward(np_images, np_hprev)
    np_pred = predict(np_outputs)

    np_total += labels.shape[0]
    np_correct += (np_pred == labels).sum().item()

print('np Accuracy of the model on the 10000 test images: {} %'.format(100 * np_correct / np_total))