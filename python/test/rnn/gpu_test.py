import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
import cpp as cpp

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

gpu_iter_loss = 0

np_U = xavier_init(hidden_size, input_size, fc=True)
np_W = xavier_init(hidden_size, hidden_size, fc=True)
np_V = xavier_init(hidden_size, hidden_size, fc=True)
np_FC_W = xavier_init(num_classes, hidden_size, fc=True)
U = cpp.cppTensor(np_U.reshape(hidden_size, input_size))
W = cpp.cppTensor(np_W.reshape(hidden_size, hidden_size))
V = cpp.cppTensor(np_V.reshape(hidden_size, hidden_size))
FC_W = cpp.cppTensor(np_FC_W)

class RNN(cpp.cppModule):
    def __init__(self, input_size, hidden_size):
        super().__init__() 

        self.hidden_size = hidden_size
        self.rnn = cpp.cppRnn(learning_rate, U, W, V, FC_W, seq_length, input_size, hidden_size, num_classes)
        self.rnn.cuda()

    def forward(self, x):
        hprev = np.zeros((self.hidden_size, 1))
        hprev = cpp.cppTensor(hprev)
        hprev.cuda()

        return self.rnn.forward(x, hprev)
    
    def parameters(self):
        return self.rnn.parameters()

    def modules(self):
        return self.rnn

gpu_model = RNN(input_size, hidden_size)
optimizer = cpp.cppAdagrad(gpu_model.parameters(), learning_rate)
criterion = cpp.cppCrossEntropyLoss()
start_time = time.time()

print("start train gpu hidden_size {}".format(hidden_size))
for epoch in range(num_epochs):
    for i, (train_images, train_labels) in enumerate(train_loader):
        train_images = train_images.reshape(seq_length, batch_size, input_size).detach().numpy()
        train_labels = train_labels.detach().numpy()

        gpu_images = [cpp.cppTensor(train_images[j]) for j in range(len(train_images))]
        gpu_labels = cpp.cppTensor(train_labels)

        [gpu_images[j].cuda() for j in range(len(gpu_images))]
        gpu_labels.cuda()
        
        gpu_outputs = gpu_model.forward(gpu_images)
        loss = criterion(gpu_outputs, gpu_labels)
        optimizer.zero_grad()
        criterion.backward(gpu_model.modules())
        optimizer.step()
        
        loss.cpu()
        gpu_iter_loss += np.sum(loss.numpy())

        if (i + 1) % interval == 0:
            print("gpu epoch {}/{} iter {}/{} loss {:.4f}".format(epoch + 1, num_epochs, i + 1, total_step, gpu_iter_loss / interval))
            print("elased time {}".format(time.time() - start_time))
            start_time = time.time()
            gpu_iter_loss = 0

# gpu_correct = 0
# gpu_total = 0

# def softmax(x):
#     e = np.exp(x)
#     return e / np.sum(e)

# def predict(outputs):
#     return np.argmax(softmax(outputs), 0)

# gpu_model.cpu()

# for test_images, test_labels in test_loader:
#     np_images = test_images.reshape(seq_length, batch_size, input_size).detach().numpy()
#     np_hprev = np.zeros((hidden_size, 1))
#     labels = test_labels.detach().numpy()

#     gpu_images = [cpp.cppTensor(np_images[j]) for j in range(len(np_images))]
#     gpu_hprev = cpp.cppTensor(np_hprev)
#     gpu_outputs = cpp.cppTensor(np.zeros((num_classes, 1)))
    
#     gpu_model.forward(gpu_outputs, gpu_images, gpu_hprev)
#     gpu_pred = predict(gpu_outputs.numpy())

#     gpu_total += labels.shape[0]
#     gpu_correct += (gpu_pred == labels).sum().item()

# print('gpu Accuracy of the model on the 10000 test images: {} %'.format(100 * gpu_correct / gpu_total))

# add test
# x = cpp.cppTensor(np.ones((128, 130)))
# x.cuda()
# y = cpp.cppTensor(np.ones((128, 130)))
# y.cuda()
# result = cpp.cppTensor(np.zeros((128, 130)))
# result.cuda()

# cpp.add_gpu(result, x, y)
# result.cpu()
# print(result.numpy())


# transpose test
# x = cpp.cppTensor(np.random.randn(128, 130))
# x_t = cpp.cppTensor(130, 128)
# x.cuda()
# x_t.cuda()
# cpp.transpose_gpu(x_t, x)
# x.cpu()
# x_t.cpu()
# print(x.numpy())
# print(x_t.numpy())

# zeros test
# x = cpp.cppTensor(np.ones((128, 130)))
# x.cuda()
# x.zeros()
# x.cpu()
# print(x.numpy())

# matMul test
# x = cpp.cppTensor(np.ones((512, 257)))
# x.cuda()
# y = cpp.cppTensor(np.ones((257, 512)))
# y.cuda()
# result = cpp.cppTensor(np.zeros((512, 512)))
# result.cuda()

# start_time = time.time()
# for i in range(10000):
#     cpp.matMul_gpu(result, x, y, False)
# print("time : {}".format(time.time() - start_time))

# start_time = time.time()
# for i in range(10000):
#     cpp.matMul_gpu(result, x, y, True)
# print("time : {}".format(time.time() - start_time))

# result.cpu()
# print(result.numpy())
