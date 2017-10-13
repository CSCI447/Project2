from NeuralNetwork import *
from NeuralNetwork.FeedForward.FF import NN
import csv
import codecs

inputArray = []
expectedOutputArray = []
with codecs.open('NeuralNetwork/Data/2_dim.csv', 'r', encoding='utf-8') as inputcsvfile:
    csv_input = csv.reader(inputcsvfile, delimiter=",")
    for row in csv_input:
        inputArray.append(row)

with codecs.open('NeuralNetwork/Data/2_dim_out.csv', 'r', encoding='utf-8') as outputcsvfile:
    csv_output = csv.reader(outputcsvfile, delimiter=",")
    for row in csv_output:
        expectedOutputArray.append(row)

hidden_layer_amount = 1
hidden_nodes_amount = 2
output_nodes_amount = 1

feedforward = NN(inputArray, expectedOutputArray, hidden_layer_amount, hidden_nodes_amount, output_nodes_amount)

feedforward.initialize()
feedforward.train(3)