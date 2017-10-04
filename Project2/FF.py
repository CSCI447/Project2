from math import exp
from random import seed
from random import random

class Neuron:
    def __init__(self):
        self.weight = 0
        self.value = 0
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

    #def updateWeights(self, value):


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

    def setValue(self, value):
        self.value = value

    def setWeight(self, value):
        self.weight = value

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

        # Initialize a network

    def initialize_network(self, n_inputs, n_hidden, n_outputs):


    def build_network(self):
        for x in self.inputs:
            n = Neuron()
            n.setValue(x)
            self.startingNodes.append(n)

        for x in self.hidden:
            n = Neuron()
            n.setValue(x)
            self.hiddenNodes.append(n)

        for x in self.outputs:
            n = Neuron()
            n.setValue(x)
            self.outputNodes.append(n)

    def connect_network(self):
        for x in self.startingNodes:
            x.setOutputNodes(self.hiddenNodes)  #prob come back and init more stuff

        for x in self.hiddenNodes:
            x.setOutputNodes(self.outputNodes)
            x.setInputNodes(self.startingNodes)

        for x in self.outputNodes:
            x.setInputNodes(self.hiddenNodes)



