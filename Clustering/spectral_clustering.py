import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# reference: https://rahuljain788.medium.com/implementing-spectral-clustering-from-scratch-a-step-by-step-guide-9643e4836a76

class Spectral:
    def __init__(self, k=2):
        self.k = k
        self.similarity_matrix = None
        self.degree_matrix = None
        self.laplacian = None
        self.labels = None
    def _make_graph(self, X, Y):
        points = np.array(list(zip(X,Y)), dtype=np.float32)
        self.similarity_matrix = np.zeros((len(points), len(points)), dtype=np.float32)
        # sigma_2 = points.std()**2
        for i in range(len(points)):
            for j in range(len(points)):
                p1, p2 = points[i], points[j]
                # self.similarity_matrix[i][j] = np.exp((-(p1-p2)**2)/sigma_2)
                self.similarity_matrix[i][j] = np.exp((-(p1-p2)**2).sum())
    def _make_laplacian(self):
        self.degree_matrix = np.diag(self.similarity_matrix.sum(axis=1))
        self.laplacian = self.degree_matrix - self.similarity_matrix
    def fit(self, X, Y):
        self._make_graph(X,Y)
        self._make_laplacian()
        eigenvalues, eigenvectors = np.linalg.eigh(self.laplacian)
        k_eigenvectors = eigenvectors[:, :self.k] # column vectors
        kmeans = KMeans(n_clusters=self.k)
        kmeans.fit(k_eigenvectors)
        self.labels = kmeans.labels_

if __name__ == "__main__":
    r1 = 4
    r2 = 8
    n = 1000
    x1 = np.random.choice([-1, 1], n)*np.random.rand(n)*r1
    x2 = np.random.choice([-1, 1], n)*np.random.rand(n)*r2
    
    y1 = np.random.choice([-1, 1], n)*((r1**2-x1**2)**0.5)
    y2 = np.random.choice([-1, 1], n)*((r2**2-x2**2)**0.5)

    X = np.array([*x1, *x2])
    Y = np.array([*y1, *y2])

    model = Spectral(k=2)
    model.fit(X, Y)
    X_0 = X[model.labels==0]; Y_0 = Y[model.labels==0]
    X_1 = X[model.labels==1]; Y_1 = Y[model.labels==1]

    plt.scatter(X_0, Y_0)
    plt.scatter(X_1, Y_1)
    plt.show()
