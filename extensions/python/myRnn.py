import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys
import numpy as np
import cpp as cpp

np.set_printoptions(threshold=sys.maxsize) 

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

def cpp_zeros_like(input):
    return ncpp.zeros_like(ncpp.NdArray(input)).getNumpyArray()

def cpp_dot(inArray1, inArray2):
    inArray1 = ncpp.NdArray(inArray1)
    inArray2 = ncpp.NdArray(inArray2)
    ncpp.memoryFree(inArray1)
    ncpp.memoryFree(inArray2)

    result = ncpp.dot(inArray1, inArray2)
    output = result.getNumpyArray()

    ncpp.memoryFree(result)
    return output

def cpp_dot_debug(inArray1, inArray2):
    inArray1 = ncpp.NdArray(inArray1)
    inArray2 = ncpp.NdArray(inArray2)
    return ncpp.dot_debug(inArray1, inArray2).getNumpyArray()

def xavier_init(c1, c2, w=1, h=1, fc=False):
    fan_1 = c2 * w * h
    fan_2 = c1 * w * h
    ratio = np.sqrt(6.0 / (fan_1 + fan_2))
    params = ratio * (2 * np.random.random((c1, c2, w, h)) - 1)
    if fc:
        params = params.reshape(c1, c2)
    return params

class My_RNN(object):
    def __init__(self, input_size, hidden_size, num_classes):
        self.lr = learning_rate
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.U = xavier_init(hidden_size, input_size, fc=True)
        self.W = xavier_init(hidden_size, hidden_size, fc=True)
        self.V = xavier_init(hidden_size, hidden_size, fc=True)
        self.b = np.zeros((hidden_size, 1))
        self.c = np.zeros((hidden_size, 1))

        self.FC_W = xavier_init(num_classes, hidden_size, fc=True)
        self.fc_b = np.zeros((num_classes, 1))

        self.mU = cpp_zeros_like(self.U)
        self.mW = cpp_zeros_like(self.W)
        self.mV = cpp_zeros_like(self.V)
        self.mb = cpp_zeros_like(self.b)
        self.mc = cpp_zeros_like(self.c)

        self.mFC_W = cpp_zeros_like(self.FC_W)
        self.mfc_b = cpp_zeros_like(self.fc_b)

        self.X = {}
        self.A = {}
        self.S = {}
        self.O = {}
        self.FC_O = {}

    def forward(self, x, hprev):
        self.S[-1] = np.copy(hprev)

        for t in range(self.seq_length):
            self.X[t] = x[t].T
            self.A[t] = self.U @ self.X[t] + self.W @ self.S[t - 1] + self.b
            self.S[t] = np.tanh(self.A[t])
            self.O[t] = self.V @ self.S[t] + self.c
        
        self.FC_O = self.FC_W @ self.O[self.seq_length - 1] + self.fc_b

        return self.FC_O

    def forward_cpp(self, x, hprev):
        self.S[-1] = np.copy(hprev)

        for t in range(self.seq_length):
            self.X[t] = x[t].T
            self.A[t] = cpp_dot(self.U, self.X[t]) + cpp_dot(self.W, self.S[t - 1]) + self.b
            self.S[t] = np.tanh(self.A[t])
            self.O[t] = cpp_dot(self.V, self.S[t]) + self.c

        self.FC_O = cpp_dot(self.FC_W, self.O[self.seq_length - 1]) + self.fc_b

        return self.FC_O

    def backward(self, dY):
        dFC_W = cpp_zeros_like(self.FC_W)
        dfc_b = cpp_zeros_like(self.fc_b)

        dU, dW, dV = cpp_zeros_like(self.U), cpp_zeros_like(self.W), cpp_zeros_like(self.V)
        db, dc = cpp_zeros_like(self.b), cpp_zeros_like(self.c)
        dS_next = cpp_zeros_like(self.S[0])

        dFC_W = dY @ self.O[self.seq_length - 1].T
        dfc_b = dY
        dO = self.FC_W.T @ dY

        dV = dO @ self.S[self.seq_length - 1].T
        dc = dO

        # np.savetxt("backward_dFC_W_T", dFC_W.T)
        # np.savetxt("backward_dY", dY)
        # np.savetxt("backward_dO", dO)

        for t in reversed(range(self.seq_length)):
            dS = self.V.T @ dO + dS_next
            # np.savetxt("backward_{}".format(t), dS)
            dA = (1 - self.S[t] ** 2) * dS
            dU += dA @ self.X[t].T
            temp = dA @ self.S[t - 1].T
            # np.savetxt("backward_{}".format(t), temp)
            dW += dA @ self.S[t - 1].T
            db += dA
            dS_next = self.W.T @ dA

        return [dU, dW, dV, db, dc, dFC_W, dfc_b]

    def backward_cpp(self, dY):
        dFC_W = cpp_zeros_like(self.FC_W)
        dfc_b = cpp_zeros_like(self.fc_b)

        dU, dW, dV = cpp_zeros_like(self.U), cpp_zeros_like(self.W), cpp_zeros_like(self.V)
        db, dc = cpp_zeros_like(self.b), cpp_zeros_like(self.c)
        dS_next = cpp_zeros_like(self.S[0])

        dFC_W = cpp_dot(dY, self.O[self.seq_length - 1].T)
        dfc_b = dY
        dO = cpp_dot(self.FC_W.T, dY)

        dV = cpp_dot(dO, self.S[self.seq_length - 1].T)
        dc = dO

        # np.savetxt("backward_cpp_dFC_W_T", dFC_W.T)
        # np.savetxt("backward_cpp_dY", dY)
        # np.savetxt("backward_cpp_dO", dO)

        # ncpp.NdArray(dFC_W.T).print()
        # ncpp.NdArray(dY).print()
        
        for t in reversed(range(self.seq_length)):
            dS = cpp_dot(self.V.T, dO) + dS_next
            # np.savetxt("backward_cpp_{}".format(t), dS)
            dA = (1 - self.S[t] ** 2) * dS
            dU += cpp_dot(dA, self.X[t].T)
            temp = cpp_dot(dA, self.S[t - 1].T)
            # np.savetxt("backward_cpp_{}".format(t), temp)
            dW += cpp_dot(dA, self.S[t - 1].T)
            db += dA
            dS_next = cpp_dot(self.W.T, dA)

        return [dU, dW, dV, db, dc, dFC_W, dfc_b]

    def optimizer_step(self, gradients):
        for dparam in gradients:
            np.clip(dparam, -5, 5, out=dparam)
            
        for param, dparam, mem in zip([self.U, self.W, self.V, self.b, self.c, self.FC_W, self.fc_b], 
                                      gradients,
                                      [self.mU, self.mW, self.mV, self.mb, self.mc, self.mFC_W, self.mfc_b]):
            mem += dparam * dparam
            param += -self.lr * dparam / np.sqrt(mem + 1e-8)
    
    def cross_entropy_loss(self, outputs, labels):
        Y = self.softmax(outputs)
        loss = -np.log(Y) * self.one_hot_vector(Y, labels)
        return Y, loss

    def softmax(self, x):
        e = np.exp(x)
        return e / np.sum(e)

    def deriv_softmax(self, Y, labels):
        dY = np.copy(Y)
        for i in range(len(labels)):
            dY[labels[i]][i] -= 1
        return dY

    def one_hot_vector(self, Y, labels):
        out = cpp_zeros_like(Y)
        for i in range(len(labels)):
            out[labels[i]][i] = 1
        return out

    def predict(self, outputs):
        return np.argmax(self.softmax(outputs), 0)

if __name__ == "__main__":
    model = My_RNN(input_size, hidden_size, num_classes)

    total_step = len(train_loader)
    iter_loss = 0
    interval = 1000
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(seq_length, batch_size, input_size).detach().numpy()
            labels = labels.detach().numpy()

            hprev = np.zeros((hidden_size, 1))
            outputs = model.forward(images, hprev)
            # outputs = model.forward_cpp(images, hprev)
            print(labels.shape)

            Y, loss = model.cross_entropy_loss(outputs, labels)
            gradients = model.backward(model.deriv_softmax(Y, labels))
            # gradients = model.backward_cpp(model.deriv_softmax(Y, labels))
            model.optimizer_step(gradients)
            iter_loss += np.sum(loss)
            if (i + 1) % interval == 0:
                print("epoch {}/{} iter {}/{} loss {:.4f}".format(epoch + 1, num_epochs, i + 1, total_step, iter_loss / interval))
                iter_loss = 0
            break
        break
