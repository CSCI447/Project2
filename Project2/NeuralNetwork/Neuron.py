from Project2.NeuralNetwork import Connection
from math import exp

class Neuron:
    def __init__(self):
        self.value = 0              #output value
        self.func = ""              #sigmoid, linear
        self.layer = ""             #layer
        self.fromConnections = []       #array of connections coming from previous layer
        self.toConnections = []      #array of connections coming from next layer

        #self.inputNodes = []
        #self.outputNodes = []
        self.error = 0              #error calculated from backprop
        # self.pastWeights = []

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


    def getFunc(self):
        return self.func

    def getLayer(self):
        return self.layer

    def getError(self):
        return self.error

    def getFromWeights(self):
        weights = []
        for c in self.fromConnections:
            weights.append(c.getWeight())
        return weights

    def getFromConnections(self):
        return self.fromConnections

    def setFromConnections(self, value):  #current activation value of node
        self.fromConnections = value

    def getToConnections(self):
        return self.toConnections

    def setToConnections(self, value):  # current activation value of node
        self.toConnections = value

    def setValue(self, value):
        self.value = value

    def getValue(self):
        return self.value

    def setFunc(self, value):
        self.func = value

    def setLayer(self, value):
        self.layer = value

    def setNodeWeightsLength(self, size):
        self.nodeWeights = [None] * size

    def setError(self, value):
        self.error = value

    def sigmoid(self, value):
        return 1.0 / (1.0 + exp(-value))

    def activate(self, weights, inputs):  #caluculate weights * inputs=
        activation = weights[-1]
        for i in range(len(weights) - 1):
            activation += weights[i] * int(inputs[i])
        return activation

    def transfer_derivative(self, output):      #currently output is the value of the node
        return output * (1.0 - output)
