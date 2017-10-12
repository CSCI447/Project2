from NeuralNetwork.RBF import *
import random
import math

class K_Means:
    def __init__(self,input_values,clusterCount):
        self.clusterCount = clusterCount                                             #k
        self.dataPointCount = 80
        self.cutoff = 0.01  # convergence threshold                                  #fixed
        self.dataPoints = self.form_dataset(input_values)
        self.clusters = self.form_clusters()
        self.centroids = self.get_centroids()


    def form_dataset(self,input_values):
        data_points = []
        for i in range(len(input_values)):
            data_point = Data_point(input_values[i])
            data_points.append(data_point)                                           #forms list of n dimensional data points
        return data_points

    def form_clusters(self):
        initial = random.sample(self.dataPoints, self.clusterCount)                  #initially choose k random points in the data sets for the centroids and form clusters according to those centroids
        clusters = [Cluster(p) for p in initial]
        while True:
            lists = [[] for _ in clusters]                                           #holds points in each cluster
            for point in self.dataPoints:                                            # For every data point in the data set
                smallest_distance = self.get_distance(point, clusters[0].centroid)       # Get the distance between that point and the centroid of the first cluster
                clusterIndex = 0                                                     # Set the cluster this point belongs to
                for i in range(self.clusterCount - 1):                               #for the rest of the clusters
                    distance = self.get_distance(point, clusters[i + 1].centroid)        # calculate the distance of that point to each other cluster's centroid
                    if distance < smallest_distance:                                 # If it's closer to that cluster's centroid
                        smallest_distance = distance                                 #update the smallest distance
                        clusterIndex = i + 1
                lists[clusterIndex].append(point)                                    #set point to belong to that cluster that's closest
            biggest_shift = 0.0                                                      #measures the shift of the clusters
            for i in range(self.clusterCount):                                       #for each cluster
                shift = clusters[i].update(lists[i])                                 #calculate delta of the centroid's position
                print("Cluster " + str(i + 1) + " has " + str(len(lists[i])) + " data points with centroid " + str(clusters[i].centroid.coords))
                biggest_shift = max(biggest_shift, shift)                            #keep track of the biggest change
            print("-----------------------------------------------")
            if biggest_shift < self.cutoff:                                          #if the centroids have stopped moving then we have convergence
                break
        return clusters

    def get_distance(self,a,b):                                                      #euclidean distance between two n-dimensional points
        difference = 0.0
        for i in range(a.n):
            squareDifference = pow(((a.coords[i]) - b.coords[i]), 2)
            difference += squareDifference
        distance = math.sqrt(difference)
        return distance

    def get_centroids(self):
        centroids = []
        for i in range(len(self.clusters)):
            centroids.append(self.clusters[i].centroid)
        return centroids

class Cluster:
    def __init__(self, centroid):
        self.points = []                                                         #points belonging to this cluster
        self.centroid = centroid #self.calculate_centroid()                                     #center point of the cluster

    def update(self, points):                                                        #returns the shift of the centroid after updating
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculate_centroid()
        new_centroid = self.centroid
        shift = self.get_centroids_distance(old_centroid, new_centroid)
        return shift

    def get_centroids_distance(self,a,b):                                                      #euclidean distance between two n-dimensional centroids
        difference = 0.0
        for i in range(a.n):
            squareDifference = pow(((float(a.coords[i])) - b.coords[i]), 2)
            difference += squareDifference
        distance = math.sqrt(difference)
        return distance

    def calculate_centroid(self):
        numPoints = len(self.points)
        dim_array = []
        for i in range(self.points[0].n):  #for n dimensions
            sum = 0
            for j in range(len(self.points)):  #all points in the cluster
                sum  += self.points[j].coords[i]
            dim_array.append(int(sum/numPoints))   #list of all coordinates in the cluster                                                    #reformat so all dimensions are together
        centroid_coords = [dList for dList in dim_array]              #mean for each dimension
        return Data_point(centroid_coords)

class Data_point:
    def __init__(self, coords):
        self.coords = self.create_int_array(coords)
        self.n = len(coords)                                                        #dimension

    def create_int_array(self,coords):
        coordinate_list =[]
        for i in range(len(coords)):
            coordinate = int(coords[i])
            coordinate_list.append(coordinate)
        return coordinate_list
