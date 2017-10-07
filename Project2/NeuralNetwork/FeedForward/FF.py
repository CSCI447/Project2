from random import random
import math

from Project2.NeuralNetwork.Connection import Connection

from Project2.NeuralNetwork.Neuron import Neuron


class NN:

    def __init__(self, input_values, hidden_layers_amount, hidden_nodes_amount, output_nodes_amount, expected_output_values, answers, learnrate = 0.3, threshold = 1, momentum = 0.5, maximum = 0, minimum = 1000):
        self.input_values = input_values                        #values from training data
        self.hidden_layers_amount = hidden_layers_amount        #number of hidden layers
        self.hidden_nodes_amount = hidden_nodes_amount          #number of nodes in hidden layer
        self.expected_output_values = expected_output_values    #expected output for error checking
        self.output_nodes_amount = output_nodes_amount          #number of output nodes
        self.answerSet = answers                                #?
        self.learnRate = learnrate                              #learnrate duh
        self.threshold = 0.01 * threshold                       #?
        self.momentum = momentum                                #momentum function
        self.maximum = maximum                                  #?
        self.minimum = minimum                                  #?
        self.inputNodes = []                                    #list of nodes in the input layer
        self.hiddenLayers = []                                   #list of hidden Layers (because this can be 0 to 2), each of these lists contains a list of hiddenNodes (see build_network)
        self.outputNodes = []                                   #list of output nodes
        self.network = list()
        self.converged = False                                  #?


    #Build network structure (layers and nodes)
    def build_network(self):
        for x in self.input_values:
            n = Neuron()
            n.setValue(x)
            self.inputNodes.append(n)
            self.network.append(self.inputNodes)

        for x in range(self.hidden_layers_amount):
            hiddenNodes = []
            for x in range(self.hidden_nodes_amount):
                n = Neuron()
                #n.setValue(x)
                hiddenNodes.append(n)
                self.hiddenLayers.append(hiddenNodes)
                self.network.append(hiddenNodes)

        for x in range(self.output_nodes_amount):
            n = Neuron()
            #n.setValue(x)
            self.outputNodes.append(n)
            self.network.append(self.outputNodes)


    #Connect the above structure by linking the nodes in the layers correctly
    def connect_network(self):
        # for x in self.inputNodes:
        #     x.setOutputNodes(self.hiddenNodes)  #prob come back and init more stuff

        for neuron in self.hiddenLayers[0]:
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
            for n in self.hiddenLayers[0]:
                c = Connection()
                c.setFromNeuron(n)
                connections.append(c)
            neuron.setConnections(connections)
            # neuron.setInputNodes(self.hiddenNodes)
            # neuron.setNodeWeightsLength(len(neuron.getInputNodes))

    #Initialize weights for all layers except input layer (because it doesn't connect back to anything, and thus has no weight)
    def initialize_weights(self):
        iterlayers = iter(self.network)
        next(iterlayers)
        for layer in iterlayers:
            for neuron in layer:
                for c in neuron.getConnections():
                    c.setWeight = random.random()
        # for neuron in self.outputNodes:
        #     for c in neuron.getConnections():
        #         c.setWeight = random.random()

    #Sets value of each node to weighted sum of connections to neurons in the layer above, processed with sigmoid funciton
    def feedforward(self):
        outputlayer_inputs = []
        for neuron in self.hiddenLayers[0]:     #loop through the hidden layer nodes
            #activation = Activation()       #create instance of activation
            activation = neuron.activate(neuron.getWeights(), self.input_values)   #sum of inconming connections * weights            activation.activate(neuron.getConnections, self.hiddenNodes)
            neuron.setValue(neuron.sigmoid(activation))      #perform sigmoid on activation summation                                    #neuron['output'] = activation.sigmoid(activation)
            outputlayer_inputs.append(neuron.getValue())

        final_outputs = []
        for neuron in self.outputNodes:
            activation = neuron.activate(neuron.getWeights(), outputlayer_inputs)
            neuron.setValue(neuron.sigmoid(activation))
            final_outputs.append(neuron.getValue())

        return final_outputs #this needs to return the output of the output layer

    #Determines errors of neurons, then update weights based on these errors
    def backprop(self, expected):

        #output layer error = (expected output - actual output) * parital_derivative
        i = 0
        for neuron in self.outputNodes:                                                     #for every neuron in the outputNodes
            unprocessed_error = self.expected_output_values[i] - neuron.getValue            #get error w/o derivative
            error_w_pd = unprocessed_error * neuron.transfer_derivative(neuron.getValue)    #get error w derivative
            neuron.setError(error_w_pd)                                                     #set as error
            i += 1

        #update hidden-output connection weights
        for neuron in self.outputNodes:                                                                             #for every neuron in the outputNodes
            for connection in neuron.getConnections():                                                                #for every connection to the neuron
                new_weight = connection.getWeight + (self.learnRate * neuron.getError() * neuron.getValue())        #set weight like the function we talked about
                connection.setWeight(new_weight)

        #hidden layer error = (weight_k * error_j) * transfer_derivative(output)
        i = 0
        for neuron in self.hiddenLayers[0]:                                                                     #for every neuron in hidden nodes
            for connect in neuron.getConnections[i]:                                                        #for all the connections to that neuron, iteresting that I didn't use connect in following code, although I don't think we need it
                unprocessed_error = self.outputNodes[i].getError() * self.outputNodes[i].getWeight()        #start at first hidden node, iterate over connections, sum with error and connection weights
                error_w_pd += unprocessed_error * neuron.transfer_derivative(neuron.getValue)               #this might not be correct actually, but hopefully you get the idea
        neuron.setError(error_w_pd)
        i += 1

        #update input-hidden connection weights
        for neuron in self.hiddenLayers[0]:                                                                             #for all neurons in hiddenNodes
            for connection in neuron.getConnections:                                                                #for all connections to that neuron
                new_weight = connection.getWeight + (self.learnRate * neuron.getError() * neuron.getValue())        #set weight like the function we talked about
                connection.setWeight(new_weight)


def train(NN, train, epochs, outputs):
        for epoch in range(epochs):
            sum_error = 0
            for row in train:
                output_values = NN.feedforward()
                expected = [0 for i in range(outputs)]
                expected[row[-1]] = 1
                sum_error += sum([(expected[i] - output_values[i]) ** 2 for i in range(len(expected))])
                NN.backprop(expected)
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, NN.learnRate, sum_error))





# class Activation:       #currently implementing activation in Neuron class, may change down the road
#     def __init__(self):
#
#     def sigmoid(self, value):
#         return 1.0 / (1.0 + exp(-value))
#
#     def activate(self, weights, current_layer):
#         activation = weights[-1]
#         for i in range(len(weights) - 1):
#             activation += weights[i] * current_layer[i]
#         return activation