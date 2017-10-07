
def evaluate(vector):
    rosenbrock = 0
    for i in range(len(vector) - 1):
        rosenbrock += (((1 - vector[i]) ** 2) + 100 * ((vector[i + 1] - (vector[i] ** 2)) ** 2))
    outfile.write(str(rosenbrock))
    outfile.write('\n')

def main():
    vector = []
    global outfile
    outfile = open("NeuralNetwork/Data/6_dim_out.csv", 'w')
    with open("NeuralNetwork/Data/6_dim.csv", 'r') as file:
        for line in file:
            currentline = line.split(",")
            vector = []
            for dim in currentline:
                vector.append(int(dim))
            evaluate(vector)
    file.close()

if __name__ == '__main__': main()