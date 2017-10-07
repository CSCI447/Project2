from random import random

from numpy import linalg

import math

from NeuralNetwork.Connection import Connection

from NeuralNetwork.Neuron import Neuron

from NeuralNetwork.RBF import K_Means

class NN:

    def __init__(self, input_values, gaussian_amount, output_nodes_amount, expected_output_values, answers, learnrate = 0.3, threshold = 1, momentum = 0.5, maximum = 0, minimum = 1000):
        self.input_values = input_values
        self.hidden_layers_amount = 1
        self.gaussian_amount = gaussian_amount
        self.expected_output_values = expected_output_values
        self.output_nodes_amount = output_nodes_amount
        self.answerSet = answers
        self.learnRate = learnrate
        self.threshold = 0.01 * threshold
        self.momentum = momentum
        self.maximum = maximum
        self.minimum = minimum
        self.inputNodes = []
        self.hiddenNodes = []
        self.outputNodes = []
        self.centroids = self.get_centroids(self.input_values, self.gaussian_amount)
        self.converged = False
        self.batches = self.form_batches()
        self.beta = 1.0

        # Initialize a network

    def build_network(self):
        for x in self.input_values:
            n = Neuron()
            n.setValue(x)
            self.inputNodes.append(n)

        for x in range(self.gaussian_amount):
            n = Neuron()
            self.hiddenNodes.append(n)

        for x in range(self.output_nodes_amount):
            n = Neuron()
            self.outputNodes.append(n)

    def connect_network(self):
        # for x in self.inputNodes:
        #     x.setOutputNodes(self.hiddenNodes)  #prob come back and init more stuff

        for neuron in self.hiddenNodes:
            connections = []
            for n in self.inputNodes:
                c = Connection()
                c.setFromNeuron(n)
                connections.append(c)
            neuron.setConnections(connections)
            #neuron.setOutputNodes(self.outputNodes)
            #neuron.setInputNodes(self.inputNodes)
            #neuron.setNodeWeightsLength(len(neuron.getInputNodes))

        for neuron in self.outputNodes:
            connections = []
            for n in self.hiddenNodes:
                c = Connection()
                c.setFromNeuron(n)
                connections.append(c)
            neuron.setConnections(connections)
            # neuron.setInputNodes(self.hiddenNodes)
            # neuron.setNodeWeightsLength(len(neuron.getInputNodes))

    def initialize_weights(self):
        #for neuron in self.hiddenNodes:    #there's only one set of weights in RBF's between the hidden layer and the output
         #   for c in neuron.getConnections():
          #      c.setWeight = random.random()
        for neuron in self.outputNodes:
            for c in neuron.getConnections():
                c.setWeight = random.random()

    def get_centroids(self,input_values,k):
        self.centroids = K_Means.K_Means(input_values,k).centroids
        return self.centroids

    def form_batches(self):
        batch_size = 10
        index = 0
        for i in range(len(self.input_values)/batch_size):
            batch = []
            for j in range(batch_size):
                batch.append(self.input_values[index])
            self.batches.append(batch)
        return self.batches

    def forward_prop(self):
        for i in self.batches:
            for j in self.batches[i]:
                for n in range(self.gaussian_amount):
                    value = self.apply_gaussian(self.batches[i][j],self.centroids[n])
                    self.hiddenNodes[n].setValue(value)

    def calculate_distance(self,x,mu):
        return linalg.norm(x-mu)

    def apply_gaussian(self,x,mu):
        phi = math.exp(-self.beta * math.pow(self.calculate_distance(x,mu),2))
        return phi

    def calculate_weighted_sum(self):
        return

    def calculate_error(self):
        return

    def update_weights(self):
        return