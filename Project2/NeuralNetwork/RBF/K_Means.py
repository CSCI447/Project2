from NeuralNetwork.RBF import *
import random
import math

class K_Means:
    def __init__(self,input_values,clusterCount):
        self.clusterCount = clusterCount                                             #k
        self.dataPointCount = 80                                                     #fixed
        self.dataPoints = self.form_dataset(self,input_values)
        self.clusters = self.form_clusters(self)
        self.centroids = self.get_centroids(self)
        self.cutoff = 0.2  # convergence threshold

    def form_dataset(self,input_values):
        data_point = Data_point()
        data_points = []
        for i in input_values:
            data_point.coords = input_values[i]
            data_points.append(data_point)                                           #forms list of n dimensional data points
        return data_points

    def form_clusters(self):
        initial = random.sample(self.dataPoints, self.clusterCount)                  #initially choose k random points in the data sets for the centroids and form clusters according to those centroids
        clusters = [Cluster([p]) for p in initial]
        while True:
            lists = [[] for _ in clusters]                                           #holds points in each cluster
            for p in self.dataPoints:                                                # For every data point in the data set
                smallest_distance = self.get_distance(p, clusters[0].centroid)       # Get the distance between that point and the centroid of the first cluster
                clusterIndex = 0                                                     # Set the cluster this point belongs to
                for i in range(self.clusterCount - 1):                               #for the rest of the clusters
                    distance = self.get_distance(p, clusters[i + 1].centroid)        # calculate the distance of that point to each other cluster's centroid
                    if distance < smallest_distance:                                 # If it's closer to that cluster's centroid
                        smallest_distance = distance                                 #update the smallest distance
                        clusterIndex = i + 1
                lists[clusterIndex].append(p)                                        #set p to belong to that cluster that's closest
            biggest_shift = 0.0                                                      #measures the shift of the clusters
            for i in range(self.clusterCount):                                       #for each cluster
                shift = clusters[i].update(lists[i])                                 #calculate delta of the centroid's position
                biggest_shift = max(biggest_shift, shift)                            #keep track of the biggest change
            if biggest_shift < self.cutoff:                                          #if the centroids have stopped moving then we have convergence
                break
        return clusters

    def get_distance(self,a,b):                                                      #euclidean distance between two n-dimensional points
        difference = 0.0
        for i in range(a.n):
            squareDifference = pow((a.coords[i] - b.coords[i]), 2)
            difference += squareDifference
        distance = math.sqrt(difference)
        return distance

    def get_centroids(self):
        centroids = []
        for i in self.clusters:
            centroids.append(self.clusters[i].centroid)
        return centroids

class Cluster:
    def __init__(self, points):
        if len(points) == 0:                                                         #check if cluster is empty
            raise Exception("ERROR: Empty Cluster!")
        self.points = points                                                         #points belonging to this cluster
        self.n = points[0].n                                                         #dimensionality of each data point
        for p in points:                                                             #make sure all points have the same dimension
            if p.n != self.n:
                raise Exception("ERROR: Mismatched Dimensions!")
        self.centroid = self.calculateCentroid()                                     #center point of the cluster

    def update(self, points):                                                        #returns the shift of the centroid after updating
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculate_centroid()
        shift = K_Means.get_distance(old_centroid, self.centroid)
        return shift

    def calculate_centroid(self):
        numPoints = len(self.points)
        coords = [p.coords for p in self.points]                                    #list of all coordinates in the cluster
        unzipped = zip(*coords)                                                     #reformat so all dimensions are together
        centroid_coords = [math.fsum(dList)/numPoints for dList in unzipped]        #mean for each dimension
        return Data_point(centroid_coords)

class Data_point:
    def __init__(self, coords):
        self.coords = coords
        self.n = len(coords)                                                        #dimension
