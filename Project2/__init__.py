from NeuralNetwork import *
from NeuralNetwork.FeedForward.FF import NN
import csv
import codecs
from Project2.Rosenbrock import generate_input_list, generate_output_list

input_list = generate_input_list(2, -3, 3, 100)
output_list = generate_output_list(input_list)

inputArray = []
expectedOutputArray = []
with codecs.open('../2_dim_input.csv', 'r', encoding='utf-8') as inputcsvfile:
    csv_input = csv.reader(inputcsvfile, delimiter=",")
    for row in csv_input:
        inputArray.append(row)


with codecs.open('../2_dim_output.csv', 'r', encoding='utf-8') as outputcsvfile:
    csv_output = csv.reader(outputcsvfile, delimiter=",")
    for row in csv_output:
        expectedOutputArray.append(row)

hidden_layer_amount = 1
hidden_nodes_amount = 4
output_nodes_amount = 1

feedforward = NN(inputArray, expectedOutputArray, hidden_layer_amount, hidden_nodes_amount, output_nodes_amount)

feedforward.initialize()
feedforward.train(1)