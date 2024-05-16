import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import load_iris

iris = load_iris()
data = iris.data

def distances_count(data, centroids):
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    return distances

def update_clusters(data, centroids):
    distances = distances_count(data, centroids)
    clusters = np.argmin(distances, axis=0)
    return clusters

def update_centroids(data, clusters, num_clusters):
    centroids = np.array([data[clusters == k].mean(axis=0)
    for k in range(num_clusters)])
    return centroids

def vizualization(data, centroids_history, clusters_history, num_clusters):
    fig, ax = plt.subplots()
    ax.set_title('K-means Clustering')

    def update(frame):
        ax.clear()
        ax.scatter(data[:, 0], data[:, 1], c=clusters_history[frame], cmap='rainbow')
        ax.scatter(centroids_history[frame][:, 0], centroids_history[frame][:, 1], c='black', marker='x', s=100)
        ax.set_title(f'Iteration {frame+1}')

    ani = FuncAnimation(fig, update, frames=len(centroids_history), repeat=False)
    plt.show()

def kmeans(data, num_clusters, max_iterations=100):
    centroids = data[np.random.choice(len(data), num_clusters, replace=False)]
    centroids_history = [centroids.copy()]
    clusters_history = []

    for i in range(max_iterations):
        clusters = update_clusters(data, centroids)
        centroids = update_centroids(data, clusters, num_clusters)
        centroids_history.append(centroids.copy())
        clusters_history.append(clusters.copy())

        if np.array_equal(centroids_history[-1], centroids_history[-2]):
            break

    vizualization(data, centroids_history, clusters_history, num_clusters)

kmeans(data[:, :2], 3)
