import numpy as np
from neural_network import NeuralNetwork
import logging

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT
)

def main():
    # initialize the network with weights
    neural_network = NeuralNetwork(w1=0.1, w2=0.2, w3=0.5, 
                               w4=0.1, w5=0.2, w6=0.1, 
                               w7=0.1, w8=0.2, w9=0.3, 
                               b1=0, b2=0, b3=0, b4=0, 
                               eta=0.01)

    np.random.seed(42)
    x1_list = np.random.randn(100)
    x2_list = np.random.randn(100)
    epochs = 100
    final_loss = neural_network.train(x1_list, x2_list, epochs)
    logging.info(f"Final loss: {final_loss}")


if __name__ == '__main__':
    main()