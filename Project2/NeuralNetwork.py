import sys
import math
import random

class Neuron:
    def __init__(self):
        self.sigma = 0.0
        self.weight = 0.0
        self.connectionLeft = Connection()
        self.connectionRight = Connection()

class Connection:
    def __init__(self):
        self.neuronLeft = Neuron()
        self.neuronRight = Neuron()

class NN:
    def __init__(self, inputs, hidden_layers, hidden_nodes, outputs):
        network = list()
        for i in range(hidden_layers):
            hidden_layer = [{'weights': [random() for i in range(inputs + 1)]} for i in range(hidden_nodes)]
            network.append(hidden_layer)
        output_layer = [{'weights': [random() for i in range(hidden_layers + 1)]} for i in range(outputs)]
        network.append(output_layer)
        return network

    def train(self,network):
        trained_network = list()

        return trained_network

    def test(self,trained_network):