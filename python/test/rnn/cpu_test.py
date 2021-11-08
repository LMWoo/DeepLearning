import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
import cpp as cpp
from cppModule import cppModule

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

cpu_iter_loss = 0

np_U = xavier_init(hidden_size, input_size, fc=True)
np_W = xavier_init(hidden_size, hidden_size, fc=True)
np_V = xavier_init(hidden_size, hidden_size, fc=True)
np_FC_W = xavier_init(num_classes, hidden_size, fc=True)
U = cpp.cppTensor(np_U.reshape(hidden_size, input_size))
W = cpp.cppTensor(np_W.reshape(hidden_size, hidden_size))
V = cpp.cppTensor(np_V.reshape(hidden_size, hidden_size))
FC_W = cpp.cppTensor(np_FC_W)

class RNN(cppModule):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__() 

        self.hidden_size = hidden_size
        self.rnn1 = cpp.cppRnn(U, W, V, seq_length, input_size, hidden_size)
        self.fc = cpp.cppLinear(FC_W, num_classes, hidden_size)

    def forward(self, x):
        hprev = np.zeros((self.hidden_size, 1))
        hprev = cpp.cppTensor(hprev)

        out = self.rnn1.forward(x, hprev)
        return self.fc.forward(out)

    def backward(self, x):
        out = self.fc.backward(x)
        return self.rnn1.backward(out)

cpu_model = RNN(input_size, hidden_size)
optimizer = cpp.cppAdagrad(cpu_model.parameters(), learning_rate)
criterion = cpp.cppCrossEntropyLoss()

print("start train cpu hidden_size {}".format(hidden_size))
start_time = time.time()

for epoch in range(num_epochs):
    for i, (train_images, train_labels) in enumerate(train_loader):
        train_images = train_images.reshape(seq_length, batch_size, input_size).detach().numpy()
        train_labels = train_labels.detach().numpy()
        train_hprev = np.zeros((hidden_size, 1))

        cpu_images = [cpp.cppTensor(train_images[j]) for j in range(len(train_images))]
        cpu_hprev = cpp.cppTensor(train_hprev)
        cpu_labels = cpp.cppTensor(train_labels)

        cpu_outputs = cpu_model.forward(cpu_images)
        loss = criterion(cpu_outputs, cpu_labels)
        
        optimizer.zero_grad()
        cpu_model.backward(criterion.dY())
        optimizer.step()

        cpu_iter_loss += np.sum(loss.numpy())

        if (i + 1) % interval == 0:
            print("cpu epoch {}/{} iter {}/{} loss {:.4f}".format(epoch + 1, num_epochs, i + 1, total_step, cpu_iter_loss / interval))
            print("elased time {}".format(time.time() - start_time))
            start_time = time.time()
            cpu_iter_loss = 0

# cpu_correct = 0
# cpu_total = 0

# def softmax(x):
#     e = np.exp(x)
#     return e / np.sum(e)

# def predict(outputs):
#     return np.argmax(softmax(outputs), 0)

# for test_images, test_labels in test_loader:
#     np_images = test_images.reshape(seq_length, batch_size, input_size).detach().numpy()
#     np_hprev = np.zeros((hidden_size, 1))
#     labels = test_labels.detach().numpy()

#     cpu_images = [cpp.cppTensor(np_images[j]) for j in range(len(np_images))]
#     cpu_hprev = cpp.cppTensor(np_hprev)
#     cpu_outputs = cpp.cppTensor(np.zeros((num_classes, 1)))

#     cpu_model.forward(cpu_outputs, cpu_images, cpu_hprev)
#     cpu_pred = predict(cpu_outputs.numpy())

#     cpu_total += labels.shape[0]
#     cpu_correct += (cpu_pred == labels).sum().item()

# print('cpu Accuracy of the model on the 10000 test images: {} %'.format(100 * cpu_correct / cpu_total))