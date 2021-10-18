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


################# gpu test #######################
# for i in range(10):
#     x = np.random.randn(4, 3)
#     y = np.random.randn(3, 5)
#     print("start")
#     print(x @ y)
#     x = cpp.numTest(x)
#     y = cpp.numTest(y)

#     result = x.dot(y)
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
for i in range(10):
    x = np.random.randn(3, 5)
    y = np.random.randn(5, 4)
    print("start")
    print(x @ y)

    x = cpp.numTest(x)
    y = cpp.numTest(y)
    result = cpp.numTest(np.random.randn(3, 4))
    cpp.dot_cpu(result, x, y)
    print('000000000000000000000')
    result.print()
    result.cuda()
    result.cpu()
    print('000000000000000000000')
    result.print()
    print("end")

    
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