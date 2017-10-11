import random
import math

from Project2.NeuralNetwork.Connection import Connection

from Project2.NeuralNetwork.Neuron import Neuron


class NN:
    def __init__(self, input_values, expected_output_values, hidden_layers_amount, hidden_nodes_amount, output_nodes_amount, learnrate=0.3, threshold=1, momentum=0.5, maximum=0, minimum=1000):
        self.input_values = input_values  # values from training data
        self.hidden_layers_amount = hidden_layers_amount  # number of hidden layers
        self.hidden_nodes_amount = hidden_nodes_amount  # number of nodes in hidden layer
        self.expected_output_values = expected_output_values  # expected output for error checking
        self.output_nodes_amount = output_nodes_amount  # number of output nodes
        #self.answerSet = answers  # ?
        self.learnRate = learnrate  # learnrate duh
        self.threshold = 0.01 * threshold  # ?
        self.momentum = momentum  # momentum function
        self.maximum = maximum  # ?
        self.minimum = minimum  # ?
        self.inputNodes = []  # list of nodes in the input layer
        self.hiddenLayers = []  # list of hidden Layers (because this can be 0 to 2), each of these lists contains a list of hiddenNodes (see build_network)
        self.outputNodes = []  # list of output nodes
        self.network = list()  # list of ALL the layers in the network
        self.connections = []
        self.converged = False  # ?

    # Build network structure (layers and nodes)
    def build_network(self):
        for x in range(len(self.input_values[0])):
            n = Neuron()
            n.setValue(x)
            self.inputNodes.append(n)
            self.network.append(self.inputNodes)

        for x in range(self.hidden_layers_amount):
            hiddenNodes = []
            for x in range(self.hidden_nodes_amount):
                n = Neuron()
                # n.setValue(x)
                hiddenNodes.append(n)
                self.hiddenLayers.append(hiddenNodes)
                self.network.append(hiddenNodes)

        for x in range(self.output_nodes_amount):
            n = Neuron()
            # n.setValue(x)
            self.outputNodes.append(n)
            self.network.append(self.outputNodes)

    # Connect the above structure by linking the nodes in the layers correctly
    def connect_network(self):
        # for x in self.inputNodes:
        #     x.setOutputNodes(self.hiddenNodes)  #prob come back and init more stuff

        for neuron in self.hiddenLayers[0]:
            for n in self.inputNodes:
                c = Connection()
                c.setFromNeuron(n)
                c.setToNeuron(neuron)
                self.connections.append(c)
            #neuron.setFromConnections(connections)

            # neuron.setOutputNodes(self.outputNodes)
            # neuron.setInputNodes(self.inputNodes)
            # neuron.setNodeWeightsLength(len(neuron.getInputNodes))

        for neuron in self.outputNodes:
            #outputConnections = []
            for n in self.hiddenLayers[0]:
                c = Connection()
                c.setFromNeuron(n)
                c.setToNeuron(neuron)
                self.connections.append(c)
            #neuron.setFromConnections(outputConnections)
            # neuron.setInputNodes(self.hiddenNodes)
            # neuron.setNodeWeightsLength(len(neuron.getInputNodes))

    # Initialize weights for all layers except input layer (because it doesn't connect back to anything, and thus has no weight)
    def initialize_weights(self):
        for c in self.connections:
            c.setWeight(random.random())
            # for neuron in self.outputNodes:
            #     for c in neuron.getConnections():
            #         c.setWeight = random.random()

    # Sets value of each node to weighted sum of connections to neurons in the layer above, processed with sigmoid funciton
    def feedforward(self, row):
        outputlayer_inputs = []
        for neuron in self.hiddenLayers[0]:  # loop through the hidden layer nodes
            # activation = Activation()       #create instance of activation
            weights = []
            for connection in self.connections:
                if connection.getToNeuron() == neuron:
                    weights.append(connection.getWeight())
            activation = neuron.activate(weights,
                                         row)  # sum of inconming connections * weights            activation.activate(neuron.getConnections, self.hiddenNodes)
            neuron.setValue(neuron.sigmoid(
                activation))  # perform sigmoid on activation summation                                    #neuron['output'] = activation.sigmoid(activation)
            outputlayer_inputs.append(neuron.getValue())

        final_outputs = []
        for neuron in self.outputNodes:
            weights = []
            for connection in self.connections:
                if connection.getToNeuron() == neuron:
                    weights.append(connection.getWeight())
            activation = neuron.activate(weights, outputlayer_inputs)
            neuron.setValue(neuron.sigmoid(activation))
            final_outputs.append(neuron.getValue())

        return final_outputs  # this needs to return the output of the output layer

    # Determines errors of neurons, then update weights based on these errors
    def backprop(self, expected):
        self.update_error_output(expected)
        self.update_weights_output()

        self.update_error_hidden()

    def update_error_output(self, expected):

        for neuron in self.outputNodes:  # for every neuron in the outputNodes
            unprocessed_error = int(expected) - neuron.getValue()  # get error w/o derivative
            error_w_pd = unprocessed_error * neuron.transfer_derivative(neuron.getValue())  # get error w derivative
            neuron.setError(error_w_pd)  # set as error



    def update_error_hidden(self):
        # hidden layer error = (weight_k * error_j) * transfer_derivative(output)
        error_w_pd = 0
        for neuron in self.hiddenLayers[0]:  # for every neuron in hidden nodes
            #update errors for each connection
            weight = 0.0
            for connect in self.connections:  # for all the connections to that neuron
                if connect.getFromNeuron() == neuron:
                    weight = connect.getWeight()
                    unprocessed_error = connect.getToNeuron().getError() * weight  # start at first hidden node, iterate over connections, sum with error and connection weights
                    error_w_pd += unprocessed_error * neuron.transfer_derivative(neuron.getValue())
            neuron.setError(error_w_pd)
            #Update weights after error has been updated
            for connect in self.connections:  # for all connections to that neuron
                if connect.getFromNeuron() == neuron:
                    new_weight = connect.getWeight() + (self.learnRate * connect.getToNeuron().getError() * neuron.getValue())  # set weight like the function we talked about
                    connect.setWeight(new_weight)

    def update_weights_output(self):
        for neuron in self.outputNodes:  # for all neurons in outputNodes
            for connection in self.connections:  # for all connections to that neuron
                if connection.getToNeuron() == neuron:
                    new_weight = connection.getWeight() + (self.learnRate * connection.getToNeuron().getError() * neuron.getValue())  # set weight like the function we talked about
                    connection.setWeight(new_weight)


    def initialize(self):
        self.build_network()
        self.connect_network()
        self.initialize_weights()

    # train a neural network for a certain number of epochs
    def train(self, epochs):
        for epoch in range(epochs):
            sum_error = 0
            for i, row in enumerate(self.input_values):
                output_values = self.feedforward(row)
                #expected = [0 for i in range(outputs_amount)]
                #expected[row[-1]] =
                for j in range(len(self.expected_output_values[i])):
                    sum_error += sum([(int(self.expected_output_values[j][0]) - output_values[j]) ** 2])
                #sum_error += sum([(int(self.expected_output_values[i][0]) - output_values[i]) ** 2 for i in range(len(self.expected_output_values))])
                self.backprop(self.expected_output_values[i][0])
            #print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, self.learnRate, sum_error))
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, self.learnRate, sum_error))


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
