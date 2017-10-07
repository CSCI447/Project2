from NeuralNetwork import Connection

class Neuron:
    def __init__(self):
        self.value = 0
        self.func = ""
        self.layer = ""
        self.connections = []
        #self.inputNodes = []
        #self.outputNodes = []
        self.error = 0
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

    def getValue(self):
        return self.value

    def getLayer(self):
        return self.layer

    def getError(self):
        return self.error

    def getPastWeights(self):
        return self.pastWeights

    def getConnections(self):
        return self.connections

    def setConnections(self, value):
        self.connections = value

    def setValue(self, value):
        self.value = value

    def setFunc(self, value):
        self.func = value

    def setLayer(self, value):
        self.layer = value

    def setNodeWeightsLength(self, size):
        self.nodeWeights = [None] * size

    def setError(self, value):
        self.error = value
