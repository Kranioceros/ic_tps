from NN import NN
import numpy as np
import matplotlib.pyplot as plt
from utils import particionar, convert_to_one_dimension

def main():
    nn = NN([4,3,2,3], learning_rate=.1)
    datos = np.genfromtxt("icgtp1datos/irisbin.csv", dtype=float, delimiter=',')
    nn.Train(datos,10,.1,0)


if __name__ == "__main__":
    main()

