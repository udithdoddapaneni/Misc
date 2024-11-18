import numpy as np

class SVM:
    def __init__(self, dims:int, c:int = 0) -> None:
        self.dims = dims
        self.weight = np.random.rand(dims) # (dims,)
        self.bias = np.float32(1)
        self.C = c
        self.lr = 0.01
    def fit(self, data:np.ndarray, classes:np.ndarray, epochs = 20) -> None:
        # data is number of (elements, dims)
        # shape of classes is (elements,)
        for epoch in range(epochs):
            for x,y in zip(data, classes):
                pred = y*((self.weight.T @ x).item() + self.bias)
                if pred >= 1:
                    continue
                dw = self.weight-self.C * y * x
                self.weight = self.weight - self.lr * dw

class LogisticRegression:
    def __init__(self, dims:int, c:int = 0) -> None:
        self.dims = dims
        self.weight = np.random.rand(dims) # (dims,)
        self.bias = np.float32(1)
        self.lr = 0.01
        
    def probability(self, x:np.ndarray):
        return 1/(1 + np.exp(-(self.weight @ x + self.bias)))
    
    def fit(self, data:np.ndarray, classes:np.ndarray, epochs = 20) -> None:
        # data is number of (elements, dims)
        # shape of classes is (elements,)
        for epoch in range(epochs):
            for x,y in zip(data, classes):
                term = np.exp(-y*(self.weight @ x + self.bias))
                # calculating gradients
                dw = -y*self.weight*term/(1+term)
                db = -y*term/(1+term)
                # weight and bias updation
                self.weight = self.weight - self.lr * dw
                self.bias = self.bias - self.lr * db
