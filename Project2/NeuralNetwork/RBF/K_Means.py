from RadialBasisFunction import *
from NeuralNetwork import *

class K_Means:
    def __init__(self):
        self.clusterCount = 0
        self.dataPointCount = 0
        self.clusters = list(Cluster())
        self.dataPoints = list(DataPoint())

class Cluster:
    def __init__(self):
        self.id = 0
        self.points = list(DataPoint())
        self.centroid = DataPoint()

class DataPoint:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.clustered = 0