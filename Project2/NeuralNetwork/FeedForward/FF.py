import random
import math

from .. Connection import Connection

from .. Neuron import Neuron


class NN:
    def __init__(self, input_values, expected_output_values, hidden_layers_amount, hidden_nodes_amount, output_nodes_amount, learnrate=0.00001, threshold=1, momentum=0.5, maximum=0, minimum=1000):
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
        self.network = []  # list of ALL the layers in the network
        self.connections = []
        self.converged = False  # check if network has converged

    # Build network structure (layers and nodes)
    def build_network(self):
        for input_neuron in range(len(self.input_values[0])):               #added [0] because self.input_values = 100, not 2
            n = Neuron()
            self.inputNodes.append(n)

        for hidden_neuron in range(self.hidden_layers_amount):
            hidden_nodes = []
            for x in range(self.hidden_nodes_amount):                       #currently only 1 hidden layer
                n = Neuron()
                hidden_nodes.append(n)
                self.hiddenLayers.append(n)                                 #TODO only appending 1 node at a time, may need to be changed
            self.network.append(hidden_nodes)                               #confirmed correct number of nodes in network

        for output_neuron in range(self.output_nodes_amount):
            n = Neuron()
            self.outputNodes.append(n)
        #self.network.append(self.outputNodes)

    # Connect the above structure by linking the nodes in the layers correctly
    def connect_network(self):
        # for x in self.inputNodes:
        #     x.setOutputNodes(self.hiddenNodes)  #prob come back and init more stuff
        for neuron in self.hiddenLayers:                             #Maybe would work with self.network[0]
            for n in self.inputNodes:
                c = Connection()
                c.setFromNeuron(n)
                c.setToNeuron(neuron)                                #TODO very much only for 1 hidden layer!  Needs to be reworked in a big way!!!!
                self.connections.append(c)
            #neuron.setFromConnections(connections)

            # neuron.setOutputNodes(self.outputNodes)
            # neuron.setInputNodes(self.inputNodes)
            # neuron.setNodeWeightsLength(len(neuron.getInputNodes))

        for neuron in self.outputNodes:
            #outputConnections = []
            for n in self.hiddenLayers:
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
            rand = random.uniform(0, 0.5)
            c.setWeight(rand)
            # for neuron in self.outputNodes:
            #     for c in neuron.getConnections():
            #         c.setWeight = random.random()


    # Sets value of each node to weighted sum of connections to neurons in the layer above, processed with sigmoid funciton
    def feedforward(self, row):
        i = 0
        for neuron in self.inputNodes:          #set values from input nodes
            neuron.setValue(row[i])
            print(neuron.getValue())
            i += 1

        for neuron in self.hiddenLayers:
            weights = []
            values = []
            for connection in self.connections:                                 #iterate to get wieghts and values=
                if connection.getToNeuron() == neuron:
                    weights.append(connection.getWeight())
                    #print(connection.getWeight())
                    values.append(connection.getFromNeuron().getValue())
                    #print(connection.getFromNeuron().getValue())
            activation = neuron.hiddenActivate(weights, values)
            print("hidden activation",activation)
            neuron.setValue(neuron.sigmoid(activation))  # perform sigmoid on activation summation
            print("hidden node value", neuron.getValue())

        for neuron in self.outputNodes:
            weights = []
            values = []
            for connection in self.connections:
                if connection.getToNeuron() == neuron:
                    weights.append(connection.getWeight())
                    print("output connection weight", connection.getWeight())
                    values.append(connection.getFromNeuron().getValue())
                    print("hidden node value", connection.getFromNeuron().getValue())
            activation = neuron.outputActivate(weights, values)
            print("output activation", activation)
            neuron.setValue(neuron.linear(activation))                                 #TODO all sigmoid baby, update when ready
            print("output node value", neuron.getValue())

        return self.outputNodes[0].getValue()



        # k = 1                                                       #keeps track of what layers we are on=
        # for layer in self.network:                                  #currently 2 layers
        #     inputs_to_layer = []
        #     for neuron in layer:                                    # loop through the hidden layer nodes
        #         weights = []
        #         for connection in self.connections:
        #             if connection.getToNeuron() == neuron:
        #                 weights.append(connection.getWeight())
        #         #print(neuron.getValue())
        #         activation = neuron.activate(weights, neuron.getValue())  # sum of inconming connections * weights
        #         if (k < self.network.__len__()):
        #             neuron.setValue(neuron.sigmoid(activation))  # perform sigmoid on activation summation                                    #neuron['output'] = activation.sigmoid(activation)
        #         else:
        #             neuron.setValue(neuron.linear(activation))
        #         inputs_to_layer.append(neuron.getValue())           #is append the correct function?
        #     previous_inputs = inputs_to_layer
        #     k += 1

        # final_outputs = []
        # for neuron in self.outputNodes:
        #     weights = []
        #     for connection in self.connections:
        #         if connection.getToNeuron() == neuron:
        #             weights.append(connection.getWeight())
        #     activation = neuron.activate(weights, outputlayer_inputs)
        #     neuron.setValue(neuron.sigmoid(activation))
        #     final_outputs.append(neuron.getValue())

        #return output_values  # this needs to return the output of the output layer

    # Determines errors of neurons, then update weights based on these errors
    def backprop(self, out_row):
        self.update_error_output(out_row)
        self.update_weights_output()

        self.update_error_hidden()
        self.update_weights_hidden()

    def update_error_output(self, out_row):
        for neuron in self.outputNodes:  # for every neuron in the outputNodes
            unprocessed_error = int(self.expected_output_values[out_row][0]) - neuron.getValue()  # get error w/o derivative
            print("output unprocessed output error = %f" % unprocessed_error)
            print("output node value: ", neuron.getValue())
            error_w_pd = unprocessed_error * neuron.linear_derivative()  # get error w derivative
            print("transfer derivative: ",neuron.linear_derivative())
            print("output error = %f" % error_w_pd)
            neuron.setError(error_w_pd)  # set as error

    def update_weights_output(self):
        for neuron in self.outputNodes:  # for all neurons in outputNodes
            for connection in self.connections:  # for all connections to that neuron
                new_weight = 0
                if connection.getToNeuron() == neuron:
                    print("update output:")
                    #print(connection.getWeight())
                    #print('learn_rate=%f' % (self.learnRate))
                    #print('error=%f' % (connection.getToNeuron().getError()))
                    #print('value=%f' % (neuron.getValue()))
                    new_weight = connection.getWeight() + (self.learnRate * connection.getToNeuron().getError() * neuron.getValue())  # set weight like the function we talked about
                    print(new_weight)
                    connection.setWeight(new_weight)
                    #print(connection.getWeight())

    def update_error_hidden(self):
        # hidden layer error = (weight_k * error_j) * transfer_derivative(output)
        for neuron in self.hiddenLayers:  # for every neuron in hidden nodes
            error_w_pd = 0
            #update errors for each connection
            weight = 0.0
            for connect in self.connections:  # for all the connections to that neuron
                if connect.getFromNeuron() == neuron:
                    weight = connect.getWeight()
                    error = connect.getToNeuron().getError()
                    unprocessed_error = error * weight  # start at first hidden node, iterate over connections, sum with error and connection weights
                    #print("hidden unprocessed error = %f" %unprocessed_error)
                    error_w_pd += unprocessed_error * neuron.transfer_derivative(neuron.getValue())
                    print("hidden error = %f" % error_w_pd)
            neuron.setError(error_w_pd)
            #Update weights after error has been updated

    def update_weights_hidden(self):
        for neuron in self.hiddenLayers:
            for connect in self.connections:  # for all connections to that neuron
                if connect.getToNeuron() == neuron:
                    print("update hidden")
                    # print(connect.getWeight())
                    # print('learn_rate=%f' % (self.learnRate))
                    # print('error=%f' % (connect.getToNeuron().getError()))
                    # print('value=%f' % (neuron.getValue()))
                    new_weight = connect.getWeight() + (self.learnRate * connect.getToNeuron().getError() * neuron.getValue())  # set weight like the function we talked about
                    print(new_weight)
                    connect.setWeight(new_weight)
                    #print(connect.getWeight())

    def initialize(self):
        self.build_network()
        self.connect_network()
        self.initialize_weights()

    # train a neural network for a certain number of epochs
    def train(self, epochs):
        for epoch in range(epochs):
        #while not(self.converged):
            sum_error = 0
            for i, row in enumerate(self.input_values):
                print("row:",i)
                output_values = self.feedforward(row)
                sum_error += (int(self.expected_output_values[i][0])-output_values)**2
                self.backprop(i)
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, self.learnRate, sum_error))

            #row2 = ['5', '5']
            #output = self.feedforward(row2)

            #error = (40016 - output)

            #print("error",error)





            # for i, row in enumerate(self.input_values):
            #     output_values = self.feedforward(row)               #magic
            #     #print(output_values)
            #     #expected = [0 for i in range(outputs_amount)]
            #     #expected[row[-1]] =
            #     # for j in range(len(self.expected_output_values)):
            #     #     sum_error += sum([(float(self.expected_output_values[j][0]) - output_values[0]) ** 2])
            #     #sum_error += sum([(int(self.expected_output_values[i][0]) - output_values[i]) ** 2 for i in range(len(self.expected_output_values))])
            #     self.backprop(self.expected_output_values[i][0])
            # #print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, self.learnRate, sum_error))
            # print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, self.learnRate, sum_error))

