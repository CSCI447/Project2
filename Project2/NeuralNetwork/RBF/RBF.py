from random import random

from numpy import linalg

import math

from Project2.NeuralNetwork.RBF.K_Means import K_Means

#import NeuralNetwork.Neuron

#import NeuralNetwork.Connection

#from Project2.NeuralNetwork.RBF import K_Means

class NN:

    def __init__(self, input_values, gaussian_amount, output_nodes_amount, expected_output_values, learnrate = 0.3, threshold = 1, momentum = 0.5, maximum = 0, minimum = 1000):
        self.input_values = input_values
        self.expected_output_values = expected_output_values
        self.training, self.testing = self.create_io_pairs(self.input_values, self.expected_output_values)
        self.hidden_layers_amount = 1
        self.gaussian_amount = gaussian_amount
        self.output_nodes_amount = output_nodes_amount
        #self.answerSet = answers
        self.learnRate = learnrate
        self.threshold = 0.01 * threshold
        self.momentum = momentum
        self.maximum = maximum
        self.minimum = minimum
        self.inputNodes = []
        self.hiddenNodes = []
        self.outputNodes = []
        self.centroids = self.get_centroids(input_values, self.gaussian_amount)
        self.converged = False
        self.batches = self.form_batches()
        self.beta = 1.0
        self.network = self.build_network(self)
        self.weighted_sum = 0
        self.squared_error = 0

        # Initialize a network

    def create_io_pairs(self,input,output):
        self.training = []
        self.testing = []
        partition = len(input) * 0.8
        for i in range(len(input)):
            if i < partition:
                ex = Example(input[i],output[i])
                self.training.append(ex)
            elif i >= partition:
                ex = Example(input[i], output[i])
                self.testing.append(ex)
        return self.training, self.testing

    def build_network(self):
        for x in self.training:
            n = Neuron()
            n.setValue(x)
            self.inputNodes.append(n)

        for x in range(self.gaussian_amount):
            n = Neuron()
            self.hiddenNodes.append(n)

        for x in range(self.output_nodes_amount):
            n = Neuron()
            self.outputNodes.append(n)
        self.connect_network(self)

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
        self.initialize_weights(self)

    def initialize_weights(self):
        #for neuron in self.hiddenNodes:    #there's only one set of weights in RBF's between the hidden layer and the output
         #   for c in neuron.getConnections():
          #      c.setWeight = random.random()
        for neuron in self.outputNodes:
            for c in neuron.getConnections():
                c.setWeight = random.random()

    def get_centroids(self,input_values,k):
        self.centroids = K_Means(input_values,k).get_centroids()
        return self.centroids

    def form_batches(self):
        batch_size = 10
        index = 0
        for i in range(len(self.training)/batch_size):
            batch = []
            for j in range(batch_size):
                batch.append(self.training[index])
            self.batches.append(batch)
        return self.batches

    def forward_prop(self):
        for i in self.batches:
            for j in self.batches[i]:
                for n in range(self.gaussian_amount):
                    value = self.apply_gaussian(self.batches[i][j],self.centroids[n]) #activation
                    self.hiddenNodes[n].setValue(value)
                self.calculate_weighted_sum(self)
                while self.calculate_error(self,batches[i],[j]) < self.threshold:
                    self.update_weights()
                print('Predicted = ' + self.weighted_sum)
                print('Actual = ' + self.batches[i][j].output)

    def calculate_distance(self,x,mu):
        return linalg.norm(x-mu)

    def apply_gaussian(self,x,mu):
        phi = math.exp(-self.beta * math.pow(self.calculate_distance(x,mu),2))
        return phi

    def calculate_weighted_sum(self):
        for i in self.hiddenNodes:
            value = self.hiddenNodes[i].value
            for j in self.outputNodes:
                self.weighted_sum += (value * self.hiddenNodes[i].connections[j].weight)
        return self.weighted_sum

    def calculate_error(self,input):
        self.squared_error = input.out - self.weighted_sum
        return self.squared_error

    def update_weights(self):
        for neuron in self.outputNodes:
            for c in neuron.getConnections():
                weight = c.getWeight()
                value = c.getFromNeuron().value
                weight = self.learnRate * self.squared_error * value
                c.weight = weight
        self.weighted_sum = self.calculate_weighted_sum(self)
        return

    def test(NN,input_values,output_values):
        return

class Example():
    def __init__(self, input, output):
        self.input = input
        self.out = output
