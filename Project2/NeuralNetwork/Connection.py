from NeuralNetwork import Neuron


class Connection:
    def __init__(self):
        self.weight = 0
        self.delta = 0
        self.fromNeuron = Neuron
        self.toNeuron = Neuron

    def getWeight(self):
        return self.weight

    def setWeight(self, value):
        self.weight = value

    def getDelta(self):
        return self.delta

    def setDelta(self, value):
        self.delta = value

    def getFromNeuron(self):
        return self.fromNeuron

    def setFromNeuron(self, value):
        self.fromNeuron = value

    def getToNeuron(self):
        return self.toNeuron

    def setToNeuron(self, value):
        self.toNeuron = value