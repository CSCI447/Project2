# handler for calling both networks
# need code for inputting the csvs into the input_values and output_values arrays

from Project2.NeuralNetwork.RBF.RBF import NN
import csv
inputArray = []
expectedOutputArray = []

with open('NeuralNetwork/Data/2_dim.csv', 'r', encoding='utf-8') as inputcsvfile:
    csv_input = csv.reader(inputcsvfile, delimiter=",")
    for row in csv_input:
        #row.insert(0,1)
        inputArray.append(row)

with open('NeuralNetwork/Data/2_dim_out.csv', 'r', encoding='utf-8') as outputcsvfile:
    csv_output = csv.reader(outputcsvfile, delimiter=",")
    for row in csv_output:
        expectedOutputArray.append(row)


rbf = NN(inputArray,expectedOutputArray,3,1)

rbf.main()