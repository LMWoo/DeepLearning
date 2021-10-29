import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

def xavier_init(c1, c2, w=1, h=1, fc=False):
    fan_1 = c2 * w * h
    fan_2 = c1 * w * h
    ratio = np.sqrt(6.0 / (fan_1 + fan_2))
    params = ratio * (2 * np.random.random((c1, c2, w, h)) - 1)
    if fc:
        params = params.reshape(c1, c2)
    return params

class npLSTM(object):
    def __init__(self, learning_rate, seq_length, x_size, hidden_size, num_classes):
        self.lr = learning_rate
        self.seq_length = seq_length
        self.input_size = x_size + hidden_size
        self.hidden_size = hidden_size
        
        self.W_f = xavier_init(hidden_size, self.input_size, fc=True)
        self.b_f = np.zeros((hidden_size, 1))
        
        self.W_i = xavier_init(hidden_size, self.input_size, fc=True)
        self.b_i = np.zeros((hidden_size, 1))
        
        self.W_g = xavier_init(hidden_size, self.input_size, fc=True)
        self.b_g = np.zeros((hidden_size, 1))
        
        self.W_o = xavier_init(hidden_size, self.input_size, fc=True)
        self.b_o = np.zeros((hidden_size, 1))
        
        self.W_fc = xavier_init(num_classes, hidden_size, fc=True)
        self.b_fc = np.zeros((num_classes, 1))
        
        self.mW_f = np.zeros_like(self.W_f)
        self.mb_f = np.zeros_like(self.b_f)
        
        self.mW_i = np.zeros_like(self.W_i)
        self.mb_i = np.zeros_like(self.b_i)
        
        self.mW_g = np.zeros_like(self.W_g)
        self.mb_g = np.zeros_like(self.b_g)
        
        self.mW_o = np.zeros_like(self.W_o)
        self.mb_o = np.zeros_like(self.b_o)
        
        self.mW_fc = np.zeros_like(self.W_fc)
        self.mb_fc = np.zeros_like(self.b_fc)
        
        self.X = {}
        self.F = {}
        self.F_A = {}
        
        self.I = {}
        self.I_A = {}
        
        self.G = {}
        self.G_A = {}
        
        self.O = {}
        self.O_A = {}
        
        self.C = {}
        self.C_A = {}
        self.H = {}
        
    def forward(self, x, hprev, cprev):
        self.X = {}
        self.F = {}
        self.F_A = {}
        
        self.I = {}
        self.I_A = {}
        
        self.G = {}
        self.G_A = {}
        
        self.O = {}
        self.O_A = {}
        
        self.C = {}
        self.C_A = {}
        self.H = {}
        
        self.H[-1] = np.copy(hprev)
        self.C[-1] = np.copy(cprev)
        
        for t in range(self.seq_length):
            self.X[t] = np.concatenate((self.H[t-1], x[t].T), axis = 0)
            
            self.F[t] = self.W_f @ self.X[t] + self.b_f
            self.F_A[t] = self.sigmoid(self.F[t])
            
            self.I[t] = self.W_i @ self.X[t] + self.b_i
            self.I_A[t] = self.sigmoid(self.I[t])
            
            self.G[t] = self.W_g @ self.X[t] + self.b_g
            self.G_A[t] = np.tanh(self.G[t])
            
            self.C[t] = self.F_A[t] * self.C[t - 1] + self.I_A[t] * self.G_A[t]
            self.C_A[t] = np.tanh(self.C[t])
            
            self.O[t] = self.W_o @ self.X[t] + self.b_o
            self.O_A[t] = self.sigmoid(self.O[t])
            
            self.H[t] = self.O_A[t] * self.C_A[t]
            
        output = self.W_fc @ self.H[self.seq_length - 1] + self.b_fc
        
        return output
    
    def backward(self, dY):
        dW_f, db_f = np.zeros_like(self.W_f), np.zeros_like(self.b_f)
        dW_i, db_i = np.zeros_like(self.W_i), np.zeros_like(self.b_i)
        dW_g, db_g = np.zeros_like(self.W_g), np.zeros_like(self.b_g)
        dW_o, db_o = np.zeros_like(self.W_o), np.zeros_like(self.b_o)
        dW_fc, db_fc = np.zeros_like(self.W_fc), np.zeros_like(self.b_fc)
        
        dH_next = np.zeros_like(self.H[0])
        dC_next = np.zeros_like(self.C[0])
        
        dW_fc = dY @ self.H[self.seq_length - 1].T
        db_fc = dY
        
        for t in reversed(range(self.seq_length)):
            dh = self.W_fc.T @ dY + dH_next
            
            dO_A = dh * self.C_A[t]
            dO = dO_A * (self.O_A[t] * (1 - self.O_A[t]))
            dW_o += dO @ self.X[t].T
            db_o += dO
            
            dC_A = self.O_A[t] * dh
            dC = dC_A * (1 - self.C_A[t] ** 2) + dC_next
            
            dF_A = dC * self.C[t - 1]
            dI_A = dC * self.G_A[t]
            dG_A = self.I_A[t] * dC
            dC_next = self.F_A[t] * dC
            
            dF = dF_A * (self.F_A[t] * (1 - self.F_A[t]))
            dW_f += dF @ self.X[t].T
            db_f += dF
            
            dI = dI_A * (self.I_A[t] * (1 - self.I_A[t]))
            dW_i += dI @ self.X[t].T
            db_i += dI
            
            dG = dG_A * (1 - self.G_A[t] ** 2)
            dW_g += dG @ self.X[t].T
            db_g += dG
            
            dX = self.W_f.T @ dF + self.W_i.T @ dI + self.W_g.T @ dG + self.W_o.T @ dO
            dH_next = dX[:self.hidden_size, :]
        
        gradients = [dW_f, db_f, dW_i, db_i, dW_g, db_g, dW_o, db_o, dW_fc, db_fc]
        
        return gradients
    
    def optimizer_step(self, gradients):
        for dparam in gradients:
            np.clip(dparam, -5, 5, out=dparam)
        
        for param, dparam, mem in zip(
            [self.W_f, self.b_f, self.W_i, self.b_i, self.W_g, self.b_g, self.W_o, self.b_o, self.W_fc, self.b_fc],
            gradients,
            [self.mW_f, self.mb_f, self.mW_i, self.mb_i, self.mW_g, self.mb_g, self.mW_o, self.mb_o, self.mW_fc, self.mb_fc]):
            mem += dparam * dparam
            param += -self.lr * dparam / np.sqrt(mem + 1e-8)
            
    def cross_entropy_loss(self, outputs, labels):
        Y = self.softmax(outputs)
        loss = -np.log(Y) * self.one_hot_vector(Y, labels)
        return Y, loss
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        e = np.exp(x)
        return e / np.sum(e)
    
    def deriv_softmax(self, Y, labels):
        dY = np.copy(Y)
        for i in range(len(labels)):
            dY[labels[i]][i] -= 1
        return dY
    
    def one_hot_vector(self, Y, labels):
        out = np.zeros_like(Y)
        for i in range(len(labels)):
            out[labels[i]][i] = 1
        return out
    
    def predict(self, outputs):
        return np.argmax(self.softmax(outputs), 0)