import random
def main():
    vector = []
    dim = 6
    with open("NeuralNetwork/Data/6_dim.csv", 'w') as file:
        for vec in range(dim * 10000):
            vector = []
            for x in range(dim):
                rand_num = random.randrange(-3,3)   # integers between 1 and 10
                vector.append(rand_num)
            file.write(str(vector)[1:-1])
            file.write("\n")
    file.close()

if __name__ == '__main__': main()