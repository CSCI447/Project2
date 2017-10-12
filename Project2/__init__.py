# handler for calling both networks
# need code for inputting the csvs into the input_values and output_values arrays

from Project2.NeuralNetwork.RBF.RBF import NN
import csv
inputTrainingArray = []
inputTestingArray = []
expectedOutputTrainingArray = []
expectedOutputTestingArray = []
with open('NeuralNetwork/Data/2_dim.csv', 'r', encoding='utf-8') as inputcsvfile:
    csv_input = csv.reader(inputcsvfile, delimiter=",")
    i = 0
    for row in csv_input:
        if(i < 80):   # training is 80%
            inputTrainingArray.append(row)
        else:
            inputTestingArray.append(row)
        i += 1
with open('NeuralNetwork/Data/2_dim_out.csv', 'r', encoding='utf-8') as outputcsvfile:
    csv_output = csv.reader(outputcsvfile, delimiter=",")
    i = 0
    for row in csv_output:
        if(i < 80):
            expectedOutputTrainingArray.append(row)
        else:
            expectedOutputTestingArray.append(row)

rbf = NN(inputTrainingArray,3,1,expectedOutputTrainingArray)