import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from npRnn import npRnn
import cpp as cpp

def xavier_init(c1, c2, w=1, h=1, fc=False):
    fan_1 = c2 * w * h
    fan_2 = c1 * w * h
    ratio = np.sqrt(6.0 / (fan_1 + fan_2))
    params = ratio * (2 * np.random.random((c1, c2, w, h)) - 1)
    if fc:
        params = params.reshape(c1, c2)
    return params

seq_length = 28
input_size = 28
hidden_size = 128
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

np_U = xavier_init(hidden_size, input_size, fc=True)
np_W = xavier_init(hidden_size, hidden_size, fc=True)
np_V = xavier_init(hidden_size, hidden_size, fc=True)
np_FC_W = xavier_init(num_classes, hidden_size, fc=True)
U = cpp.numTest(np_U.reshape(hidden_size, input_size))
W = cpp.numTest(np_W.reshape(hidden_size, hidden_size))
V = cpp.numTest(np_V.reshape(hidden_size, hidden_size))
FC_W = cpp.numTest(np_FC_W)

cpp_model = cpp.cppRnn(learning_rate, U, W, V, FC_W, seq_length, input_size, hidden_size, num_classes)
cpp_model.cuda()

np_model = npRnn(learning_rate, seq_length, hidden_size, num_classes, np_U, np_W, np_V, np_FC_W)

total_step = len(train_loader)
interval = 1000

np_iter_loss = 0
cpu_iter_loss = 0
gpu_iter_loss = 0

for epoch in range(num_epochs):
    for i, (train_images, train_labels) in enumerate(train_loader):
        np_images = train_images.reshape(seq_length, batch_size, input_size).detach().numpy()
        np_labels = train_labels.detach().numpy()
        np_hprev = np.zeros((hidden_size, 1))

        images = [cpp.numTest(np_images[j]) for j in range(len(np_images))]
        hprev = cpp.numTest(np_hprev)
        labels = cpp.numTest(np_labels)

        outputs = cpp.numTest(np.zeros((num_classes, 1)))
        Y = cpp.numTest(np.zeros((num_classes, 1)))
        dY = cpp.numTest(np.zeros((num_classes, 1)))
        loss = cpp.numTest(np.zeros((num_classes, 1)))

        ####### numpy ##########
        print('=========== start np ===========')
        np_outputs = np_model.forward(np_images, np_hprev)
        np_Y, np_loss = np_model.cross_entropy_loss(np_outputs, np_labels)
        np_dY = np_model.deriv_softmax(np_Y, np_labels)
        print(np_dY)
        print('=========== end np ===========')

        ######## cpu ##########
        print('=========== start cpu ===========')
        cpp_model.cpu()
        cpp_model.forward(outputs, images, hprev)
        cpp_model.cross_entropy_loss(dY, Y, loss, outputs, labels)
        dY.print()
        print('=========== end cpu ===========')

        ######### cuda #########
        print('=========== start gpu ===========')
        Y.zeros()
        loss.zeros()
        dY.zeros()

        cpp_model.cuda()
        [images[j].cuda() for j in range(seq_length)]
        hprev.cuda()
        outputs.cuda()
        labels.cuda()
        Y.cuda()
        dY.cuda()
        loss.cuda()

        cpp_model.forward(outputs, images, hprev)
        cpp_model.cross_entropy_loss(dY, Y, loss, outputs, labels)
        dY.print()
        print('=========== end gpu ===========')
