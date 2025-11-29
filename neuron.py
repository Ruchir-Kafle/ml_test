import numpy as np
import helpers

class Neuron:
    def __init__(self, weights: np.ndarray[int], bias: int) -> None:
        self.weights: np.ndarray[int] = weights
        self.bias: int = bias

    def feed(self, neuron_inputs: np.ndarray[int]) -> int:
        # print(self.weights, neuron_inputs)
        dot_product: np.ndarray[int] = np.dot(self.weights, neuron_inputs)
        total: int = dot_product + self.bias
        # output: int = helpers.sigmoid(total)

        return total


# Nueron testing

# weights: np.ndarray[int] = np.array([0, 1])
# bias: int = 4
# neuron_inputs: np.ndarray[int] = np.array([2, 3])

# my_neuron: Neuron = Neuron(weights, bias)
# feedforward_output: int = my_neuron.feed(neuron_inputs)
# print(feedforward_output)