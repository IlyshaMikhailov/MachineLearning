import numpy as np


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def region_query(points, point_index, eps):
    neighbors = []
    for i, point in enumerate(points):
        if i != point_index and euclidean_distance(points[point_index], point) < eps:
            neighbors.append(i)
    return neighbors


def expand_cluster(points, labels, point_index, neighbors, cluster_label, eps, min_samples):
    labels[point_index] = cluster_label
    i = 0
    while i < len(neighbors):
        neighbor_index = neighbors[i]
        if labels[neighbor_index] == -1:
            labels[neighbor_index] = cluster_label
        elif labels[neighbor_index] == 0:
            labels[neighbor_index] = cluster_label
            new_neighbors = region_query(points, neighbor_index, eps)
            if len(new_neighbors) >= min_samples:
                neighbors.extend(new_neighbors)
        i += 1


def dbscan(points, eps, min_samples):
    labels = np.zeros(len(points))  # 0 - unvisited, -1 - noise
    cluster_label = 0

    for i, point in enumerate(points):
        if labels[i] != 0:
            continue
        neighbors = region_query(points, i, eps)
        if len(neighbors) < min_samples:
            labels[i] = -1
        else:
            cluster_label += 1
            expand_cluster(points, labels, i, neighbors, cluster_label, eps, min_samples)

    return labels.astype(int)


# Пример использования:
if __name__ == '__main__':
    import pygame

    points = []
    r = 10
    minPts, eps = 4, 3 * r
    colors = ['blue', 'green', 'purple', 'red']
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    screen.fill('white')
    running = True
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    point = pygame.mouse.get_pos()
                    points.append(point)
                    pygame.draw.circle(screen, 'black', point, r)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    labels = dbscan(np.array(points), eps, minPts)
                    for i in range(len(points)):
                        pygame.draw.circle(screen, colors[labels[i]], points[i], r)
        pygame.display.flip()
    pygame.quit()
