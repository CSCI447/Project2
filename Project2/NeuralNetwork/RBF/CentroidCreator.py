import random
import math
from itertools import product

class Centroids:
    def __init__(self,dim):
        self.dim = dim
        self.centroids = []
        self.k = 0
        self.centroids = self.generate_centroid(dim)


    def generate_centroids(self, dim):
        c = Centroid()                  #create centroid
        pos = [-3,-3]                                #calculate position
        c.set_pos(pos)                     #set pos
        self.centroids.append(c)        #add to self.centroids
        d = Centroid()                  #create centroid
        pos = [3,-3]                                #calculate position
        d.set_pos(pos)                     #set pos
        self.centroids.append(d)        #add to self.centroids
        e = Centroid()                  #create centroid
        pos = [-3,3]                                #calculate position
        e.set_pos(pos)                     #set pos
        self.centroids.append(e)        #add to self.centroids
        f = Centroid()                  #create centroid
        pos = [3,3]                                #calculate position
        f.set_pos(pos)                     #set pos
        self.centroids.append(f)        #add to self.centroids
        g = Centroid()                  #create centroid
        pos = [3,0]                                #calculate posreturn self.centroids
        g.set_pos(pos)                     #set pos
        self.centroids.append(g)        #add to self.centroids   get_centroids(self):
        h = Centroid()                  #create centroid
        pos = [0,-3]                                #calculate posreturn self.centroids
        h.set_pos(pos)                     #set pos
        self.centroids.append(h)        #add to self.centroids   ntroid:
        i = Centroid()                  #create centroid
        pos = [0,0]                                #calculate posreturn
        i.set_pos(pos)                     #set pos
        self.centroids.append(i)        #add to self.centroids   ntroid:
        j = Centroid()                  #create centroid
        pos = [0,3]                                #calculate posreturn
        j.set_pos(pos)                     #set pos
        self.centroids.append(j)        #add to self.centroids   ntroid:
        k = Centroid()                  #create centroid
        pos = [3,0]                                #calculate posreturn
        k.set_pos(pos)                     #set pos
        self.centroids.append(k)        #add to self.centroids   ntroid:
        return self.centroids

    def generate_centroid(self,dim):
        range = [-3,3]
        #self.centroids = (list(tup) for tup in product(iterable, dim))
        #print(tup[0])
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