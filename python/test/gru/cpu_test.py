import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cpp as cpp

seq_length = 28
input_size = 28
hidden_size = 256
num_layers = 1
num_classes = 10
batch_size = 1
num_epochs = 1
learning_rate = 0.01
start_time = 0

for i in range(2):
    gru_model = cpp.cppGRU()
    
    train_images = np.random.randn(seq_length, batch_size, input_size)
    train_labels = np.random.randint(0, 1, (num_classes, 1))
    train_hprev = np.zeros((hidden_size, 1))
    
    gpu_images = [cpp.cppTensor(train_images[j]) for j in range(len(train_images))]
    gpu_hprev = cpp.cppTensor(train_hprev)
    gpu_labels = cpp.cppTensor(train_labels)
    gpu_outputs = cpp.cppTensor(np.zeros((num_classes, 1)))
    gpu_Y = cpp.cppTensor(np.zeros((num_classes, 1)))
    gpu_dY = cpp.cppTensor(np.zeros((num_classes, 1)))
    gpu_loss = cpp.cppTensor(np.zeros((num_classes, 1)))
    
    gru_model.forward(gpu_outputs, gpu_images, gpu_hprev)
    gru_model.cross_entropy_loss(gpu_dY, gpu_Y, gpu_loss, gpu_outputs, gpu_labels)
    gru_model.backward(gpu_dY)
    gru_model.optimizer()