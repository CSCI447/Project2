from RadialBasisFunction import *
from NeuralNetwork import *

class K_Means:
    def __init__(self,clusterCount):
        self.clusterCount = clusterCount
        self.dataPointCount = 80
        self.clusters = list(Cluster())
        self.dataPoints = list(DataPoint())

    def formClusters(self,clusterCount):

        return clusters

class Cluster:
    def __init__(self):
        self.id = 0
        self.points = list(DataPoint())
        self.centroid = DataPoint()

    def getCentroid(self):
        return self.centroid

    def getPoints(self):
        return self.points
    
    def getClusterID(self):
        return self.id

class DataPoint:
    def __init__(self,coordinates, cluster):
        self.coordinates = coordinates
        self.cluster = cluster

    def getCluster(self):
        return self.cluster

    def getCoords(self):
        return self.coordinates