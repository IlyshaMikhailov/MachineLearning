import pygame as pygame
from sklearn.cluster import DBSCAN

if __name__ == '__main__':
    points = []
    r=10
    minPts,eps = 4, 3*r
    colors = ['blue','green','purple','red']
    pygame.init()
    screen = pygame.display.set_mode((800,600))
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
                    pygame.draw.circle(screen,'black',point,r)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    dbscan = DBSCAN(eps=eps, min_samples=minPts)
                    dbscan.fit(points)
                    labels =dbscan.labels_
                    for i in range(len(points)):
                        pygame.draw.circle(screen,colors[labels[i]],points[i],r)
        pygame.display.flip()
    pygame.quit()