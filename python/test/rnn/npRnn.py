import numpy as np

class npRnn(object):
    def __init__(self, learning_rate, seq_length, hidden_size, num_classes, U, W, V, FC_W):
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
