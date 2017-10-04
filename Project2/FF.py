from math import exp
from random import seed
from random import random


class FF:
    # Initialize a network
    def initialize_network(n_inputs, n_hidden, n_outputs):
        network = list()
        hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
        network.append(hidden_layer)
        output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
        network.append(output_layer)
        return network

    def


class Neuron:
    def __init__(self):
        self.weight
        self.func = ""
        self.layer = ""
        self.inputNodes = []
        self.nodeWeights = []
        self.outputNodes = []
        self.error = 0
        self.pastWeights = []

    # def addInputs(self, nodes):
    #     for x in nodes:
    #         x.addOutput(self)
    #         self.inputs.append(x)
    #         self.weights.append(random.random())
    #         self.historicalWeights.append(0)
    #     if self.func == 'R':
    #         self.inputs.append(node(appFunc='B', value=-1))
    #     else:
    #         self.inputs.append(node(appFunc='B', value=1))
    #     self.weights.append(random.random())
    #     self.historicalWeights.append(0)

    def updateWeights(self, value):


    def getWeight(self):
        return self.weight

    def getFunc(self):
        return self.func

    def getLayer(self):
        return self.layer

    def getInputNodes(self):
        return self.inputNodes

    def getNodeWeights(self):
        return self.nodeWeights

    def getOutputNodes(self):
        return self.outputNodes

    def getError(self):
        return self.error

    def getPastWeights(self):
        return self.pastWeights

    def setWeight(self, value):
        self.Weight = value

    def setFunc(self, value):
        self.func = value

    def setLayer(self, value):
        self.layer = value

    def setInputNode(self, value):
        self.inputNode = value

    def setNodeWeights(self, value):
        self.nodeWeights = value

    def setOutputNodes(self, value):
        self.outputNodes = value

    def setError(self, value):
        self.error = value

    def set_past_weights(self, value):
        self.pastWeights = value


class NN:

    def __init__(self, inputs, hidden, outputs, answers, learnrate = 0.3, threshold = 1, momentum = 0.5, maximum = 0, minimum = 1000):
        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs
        self.answerSet = answers
        self.learnRate = learnrate
        self.threshold = 0.01 * threshold
        self.momentum = momentum
        self.maximum = maximum
        self.minimum = minimum
        self.startingNodes = []
        self.hiddenNodes = []
        self.outputNodes = []
        self.converged = False

    def set

    def build_network(self):
        for x in self.inputs:
            n = Neuron(value=x)
            self.startingNodes.append(n)

        for x in self.hidden:
            n = Neuron(value=x)
            self.hiddenNodes.append(n)
            for y in self.inputs:
                n.inputNodes(self.startingNodes[y])

        for x in self.outputs:
            n = Neuron(value=x)
            self.outputNodes.append(n)
            for y in self.hidden:
                n.inputNodes(self.startingNodes[y])

    def connect_network(self):
        for x in self.startingNodes:
            x.setOutputNodes(self.hiddenNodes)  #prob come back and init more stuff

        for x in self.hiddenNodes:
            x.setOutputNodes(self.outputNodes)
            x.setInputNodes(self.startingNodes)

        for x in self.outputNodes:
            x.setInputNodes(self.hiddenNodes)