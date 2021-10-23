import torch
import numpy as np
import cpp as cpp

# x = torch.randn(4)
# y = torch.randn(4)
# print(ncpp.sigmoid_add(x, y))

# shape = ncpp.Shape(5, 3)
# cArray = ncpp.NdArray(shape)
# ret = cArray.ones()
# cArray = ncpp.zeros_like(cArray);
# print(type(cArray));
# print(type(cArray.getNumpyArray()));
# print(cArray.getNumpyArray());
# print(ncpp.toNumCpp(cArray.getNumpyArray()));

def printResult(x, y):
    print('numpy')
    print(x @ y)

    x = cpp.NdArray(x)
    y = cpp.NdArray(y)
    print('numcpp')
    print(cpp.dot(x, y).getNumpyArray())

def dot_test():
    x = np.random.randn(5, 3)
    x = cpp.NdArray(x)
    y = np.random.randn(3, 5)
    y = cpp.NdArray(y)  

    result = cpp.dot(x, y)
    return result

# for i in range(1):
#     x = np.random.randn(5, 3)
#     y = np.random.randn(3, 5)
#     x = cpp.NdArray(x)
#     y = cpp.NdArray(y)
#     print('==============================')


# print("================================")


# xx.print()

# cpp.test_gpu()
# cpp.test_gpu_matrix_add()

# print(np.random.random((3, 4)))

# r = cpp.rnn(0.01, 28, 28, 128, 10)
# images = np.random.randn(28, 1, 28)
# hprev = np.random.randn(128, 1)
# labels = np.array([[2],])
# labels = cpp.NdArray(labels)

# print("one")
# outputs = r.forward(images, hprev)
# print("two")
# outputs = r.forward(images, hprev)

# outputs = r.forward(images, hprev)
# Y, loss = r.cross_entropy_loss(outputs, labels)
# gradients = r.backward(r.deriv_softmax(Y, labels))
# r.optimizer(gradients)

################# rnn test #######################
def xavier_init(c1, c2, w=1, h=1, fc=False):
    fan_1 = c2 * w * h
    fan_2 = c1 * w * h
    ratio = np.sqrt(6.0 / (fan_1 + fan_2))
    params = ratio * (2 * np.random.random((c1, c2, w, h)) - 1)
    if fc:
        params = params.reshape(c1, c2)
    return params

class My_RNN(object):
    def __init__(self, input_size, hidden_size, num_classes, U, W, V, FC_W):
        self.lr = learning_rate
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.U = U
        self.W = W
        self.V = V

        self.b = np.zeros((hidden_size, 1)) # rnn input parameters
        self.c = np.zeros((hidden_size, 1)) # rnn output parameters
        
        self.FC_W = FC_W
        self.fc_b = np.zeros((num_classes, 1)) # fc parameters
        
        self.mU = np.zeros_like(self.U)
        self.mW = np.zeros_like(self.W)
        self.mV = np.zeros_like(self.V)
        self.mb = np.zeros_like(self.b)
        self.mc = np.zeros_like(self.c)
        
        self.mFC_W = np.zeros_like(self.FC_W)
        self.mfc_b = np.zeros_like(self.fc_b)
        
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
            self.O[t] = self.V @ self.S[t] + self.c # (hidden, hidden) @ (hidden, 1) + (hidden, 1)
            
        self.FC_O = self.FC_W @ self.O[self.seq_length - 1] + self.fc_b # (classes, hidden) @ (hidden, 1) + (classes, 1)

        return self.FC_O # (classes, 1)
    
    def backward(self, dY): # (classes, 1)
        # zero grad
        dFC_W = np.zeros_like(self.FC_W)
        dfc_b = np.zeros_like(self.fc_b)
        
        dU, dW, dV = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
        db, dc = np.zeros_like(self.b), np.zeros_like(self.c)
        dS_next = np.zeros_like(self.S[0])
        
        dFC_W = dY @ self.O[self.seq_length - 1].T # (classes, 1) @ (1, hidden)
        dfc_b = dY # (classes, 1)
        dO = self.FC_W.T @ dY
        
        dV = dO @ self.S[self.seq_length - 1].T
        dc = dO
        
        for t in reversed(range(self.seq_length)):
            dS = self.V.T @ dO + dS_next
            dA = (1 - self.S[t] ** 2) * dS
            dU += dA @ self.X[t].T
            dW += dA @ self.S[t - 1].T
            db += dA
            dS_next = self.W.T @ dA
            
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
            dY[int(labels[i])][i] -= 1
        return dY
    
    def one_hot_vector(self, Y, labels):
        out = np.zeros_like(Y)
        for i in range(len(labels)):
            out[int(labels[i])][i] = 1
        return out
    
    def predict(self, outputs):
        return np.argmax(self.softmax(outputs), 0)
        

seq_length = 28
input_size = 28
hidden_size = 128
num_layers = 1
num_classes = 10
batch_size = 1
num_epochs = 2
learning_rate = 0.01

np_U = xavier_init(hidden_size, input_size, fc=True)
np_W = xavier_init(hidden_size, hidden_size, fc=True)
np_V = xavier_init(hidden_size, hidden_size, fc=True)
np_FC_W = xavier_init(num_classes, hidden_size, fc=True)
U = cpp.numTest(np_U.reshape(hidden_size, input_size))
W = cpp.numTest(np_W.reshape(hidden_size, hidden_size))
V = cpp.numTest(np_V.reshape(hidden_size, hidden_size))
FC_W = cpp.numTest(np_FC_W)

model = cpp.cppRnn(learning_rate, U, W, V, FC_W, seq_length, input_size, hidden_size, num_classes)
model.cuda()

np_model = My_RNN(input_size, hidden_size, num_classes, np_U, np_W, np_V, np_FC_W)

for i in range(2000000000000000000000000000000000000000000000000000000000000):
    np_images = np.random.randn(seq_length, 1, input_size)
    np_hprev = np.zeros((hidden_size, 1))
    np_labels = np.random.randint(0, num_classes, (1, ))

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
    print(np_loss)
    print('=========== end np ===========')

    ######## cpu ##########
    print('=========== start cpu ===========')
    model.cpu()
    model.forward(outputs, images, hprev)
    model.cross_entropy_loss(dY, Y, loss, outputs, labels)
    loss.print()
    print('=========== end cpu ===========')

    ######### cuda #########
    print('=========== start gpu ===========')
    Y.zeros();
    loss.zeros();
    dY.zeros();

    model.cuda()
    [images[j].cuda() for j in range(seq_length)]
    hprev.cuda()
    outputs.cuda()
    labels.cuda()
    Y.cuda()
    dY.cuda()
    loss.cuda()

    model.forward(outputs, images, hprev)
    model.cross_entropy_loss(dY, Y, loss, outputs, labels)
    loss.print()
    print('=========== end gpu ===========')

##################################################

################# gpu test #######################
# for i in range(10):
#     x = np.random.randn(4, 3)
#     y = np.random.randn(3, 5)
#     print("start")
#     print(x @ y)
#     x = cpp.numTest(x)
#     x.print_pointer()
#     y = cpp.numTest(y)
#     y.print_pointer()

#     result = cpp.numTest(np.random.randn(4, 5))
#     cpp.dot_cpu(result, x, y)
#     result.cuda()
#     result.cpu()

#     result.print()
    
#     x.cuda()
#     y.cuda()
#     gpu_result = np.random.randn(4, 5)
#     gpu_result = cpp.numTest(gpu_result)
#     gpu_result.cuda()
#     cpp.dot_gpu(gpu_result, x, y)

#     gpu_result.cpu()
#     gpu_result.print()
####################################################    
# for i in range(10):
#     x = np.random.randn(3, 5)
#     y = np.random.randn(5, 4)
#     print("start")
#     print(x @ y)

#     x = cpp.numTest(x)
#     y = cpp.numTest(y)
#     result = cpp.numTest(np.random.randn(3, 4))
#     cpp.dot_cpu(result, x, y)
#     print('000000000000000000000')
#     result.print()
#     result.cuda()
#     result.cpu()
#     print('000000000000000000000')
#     result.print()
#     print("end")

    
    # print("start")
    # x.print()
    # print('-----------')
    # x_t.print()
    # print("end")
    
    # y = np.random.randn(3, 5)

    # # print('start')
    # # print(x @ y)

    # # x = cpp.numTest(x)
    # # y = cpp.numTest(y)

    # # x.gpu_mul()
    # # result = x.dot(y)

    # # result.print()
    # # print('end')

# for i in range(1):
#     print('python print start')
#     r = cpp.rnn(0.01, 28, 28, 128, 10)
#     images = np.random.randn(28, 1, 28)
#     hprev = np.random.randn(128, 1)
#     labels = np.array([[2],])
#     labels = cpp.NdArray(labels)

#     outputs = r.forward(images, hprev)
#     outputs.useArray()
#     print('python print end')

    # Y, loss = r.cross_entropy_loss(outputs, labels)