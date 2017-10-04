from math import exp
from random import seed
from random import random
from Project2.FeedForward.Neuron import Neuron
from Project2.FeedForward.Connection import Connection

class NN:

    def __init__(self, input_values, hidden_layers_amount, hidden_nodes_amount, output_nodes_amount, expected_output_values, answers, learnrate = 0.3, threshold = 1, momentum = 0.5, maximum = 0, minimum = 1000):
        self.input_values = input_values
        self.hidden_layers_amount = hidden_layers_amount
        self.hidden_nodes_amount = hidden_nodes_amount
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
        self.converged = False

        # Initialize a network

    def build_network(self):
        for x in self.input_values:
            n = Neuron()
            n.setValue(x)
            self.inputNodes.append(n)

        for x in range(self.hidden_nodes_amount):
            n = Neuron()
            #n.setValue(x)
            self.hiddenNodes.append(n)

        for x in range(self.output_nodes_amount):
            n = Neuron()
            #n.setValue(x)
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
        for neuron in self.hiddenNodes:
            for c in neuron.getConnections():
                c.setWeight = random.random()
        for neuron in self.outputNodes:
            for c in neuron.getConnections():
                c.setWeight = random.random()

    def feedforward(self):
        x = 0

    def backprop(self):
        x = 0




