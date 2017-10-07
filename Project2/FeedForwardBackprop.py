from NeuralNetwork import *


class FF:

    def __init__(self, hidden_layers):
        self.epsilon = 0.0
        self.learn_rate = 0.1
        self.bias = Neuron()
        self.linear = True
        self.sigmoid = False

    #def initializehiddelayers(self,hidden_layers):


def feedforward(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

    #def backprop(self):


    #def updateWeights(self):

def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))
