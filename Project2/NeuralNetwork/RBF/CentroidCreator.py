import random
import math
from itertools import product

class Centroids:
    def __init__(self,dim):
        self.dim = dim
        self.centroids = []
        self.k = 0
        self.centroids = self.generate_centroid(dim)

    # evenly spacing centroids in a range of -3 to 3 for each dimension
    def generate_centroid(self,dim):
        range = [-3,0,3]
        centroidarray = list(map(list, product(range, repeat=dim)))
        for pos in centroidarray:
            centroid = Centroid()
            centroid.set_pos(pos)
            self.centroids.append(centroid)
        self.k = len(self.centroids)
        return self.centroids

    def get_centroids(self):
        return self.centroids

class Centroid:
    def __init__(self):
        self.position = []

    def set_pos(self, pos):
        self.position = pos