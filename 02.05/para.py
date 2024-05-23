from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

if __name__ == '__main__':
    iris = load_iris()
    x = iris.data
    classes = iris.target
    plt.scatter(x[:,0],x[:,1], c = classes)
    plt.show()
    pca = PCA(n_components=2)
    pca.fit_transform(x)
    print(x)