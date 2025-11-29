import numpy as np
import neuron
import data
import helpers
 
class NeuralNetwork:
    def __init__(self) -> None:
        self.h1 = neuron.Neuron(np.array([np.random.normal(), np.random.normal()]), np.random.normal())
        self.h2 = neuron.Neuron(np.array([np.random.normal(), np.random.normal()]), np.random.normal())
        self.output = neuron.Neuron(np.array([np.random.normal(), np.random.normal()]), np.random.normal())

        # self.hidden_layers = {1: [self.h1, self.h2]}


    def feedforward(self, network_inputs: np.ndarray[int]) -> list[int]: 
        # for input_1 in inputs:
        #     for input_2 in inputs:

        #         print(input)

        h1 = self.h1.feed(network_inputs)
        h2 = self.h2.feed(network_inputs)

        output = self.output.feed(np.array([helpers.sigmoid(h1), helpers.sigmoid(h2)]))

        return [h1, h2, output]
    
    def train(self, network_inputs: np.ndarray[np.ndarray[str | int]]) -> int:
        learn_rate = 0.1
        epochs = 1000


        for epoch in range(epochs):
            for sample in network_inputs:
                sample_weight: int = sample[0]
                sample_height: int = sample[1]
                sample_gender: int = sample[2]

                h1_sum, h2_sum, output_sum = self.feedforward(np.array([sample_weight, sample_height]))
                h1 = helpers.sigmoid(h1_sum)
                h2 = helpers.sigmoid(h2_sum)
                output = helpers.sigmoid(output_sum)

                # Defining derivative sigmoid values        
                h1_sigmoid_derivative = helpers.derivative_of_sigmoid(h1_sum)
                h2_sigmoid_derivative = helpers.derivative_of_sigmoid(h2_sum)
                output_sigmoid_derivative = helpers.derivative_of_sigmoid(output_sum)
                
                dL_doutput: int = -2 * (sample_gender - output)

                # Neuron output
                doutput_dw5 = h1 * output_sigmoid_derivative
                doutput_dw6 = h2 * output_sigmoid_derivative
                doutput_db3 = output_sigmoid_derivative

                doutput_dh1 = self.output.weights[0] * output_sigmoid_derivative
                doutput_dh2 = self.output.weights[1] * output_sigmoid_derivative

                # Neuron h2 
                dh2_dw3 = sample_weight * h2_sigmoid_derivative
                dh2_dw4 = sample_height * h2_sigmoid_derivative
                dh2_db2 = h2_sigmoid_derivative

                # Neuron h1 
                dh1_dw1 = sample_weight * h1_sigmoid_derivative
                dh1_dw2 = sample_height * h1_sigmoid_derivative
                dh1_db1 = h1_sigmoid_derivative

                # Updating weights & biases
                # Neuron output
                self.output.weights[0] -= learn_rate * dL_doutput * doutput_dw5
                self.output.weights[1] -= learn_rate * dL_doutput * doutput_dw6
                self.output.bias -= learn_rate * dL_doutput * doutput_db3

                # Neuron h2
                self.h2.weights[0] -= learn_rate * dL_doutput * doutput_dh2 * dh2_dw3
                self.h2.weights[1] -= learn_rate * dL_doutput * doutput_dh2 * dh2_dw4
                self.h2.bias -= learn_rate * dL_doutput * doutput_dh2 * dh2_db2

                # Neuron h1
                self.h1.weights[0] -= learn_rate * dL_doutput * doutput_dh1 * dh1_dw1
                self.h1.weights[1] -= learn_rate * dL_doutput * doutput_dh1 * dh1_dw2
                self.h1.bias -= learn_rate * dL_doutput * doutput_dh1 * dh1_db1

            if epoch % 10 == 0:
                network_predictions: list[int] = []
                input_actuals: list[int] = []
                for sample in network_inputs:
                    input_actuals.append(sample[2])

                    [*_, output] = self.feedforward(np.array([sample[0], sample[1]]))
                    network_predictions.append(helpers.sigmoid(output))
                
                loss: list[int] = helpers.mse(np.array(input_actuals), np.array(network_predictions))

                print(f"epoch {epoch}, loss {loss}")


# Dataset
dataset: np.ndarray[np.ndarray[str | int]] = np.array(data.normalize_dataset(data.data))
print(dataset)

# Neural network testing
# network_inputs: np.ndarray[int] = np.array([2, 3])

my_neural_network: NeuralNetwork = NeuralNetwork()
my_neural_network.train(dataset)

# Predict

emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
*_, emily_prediction = my_neural_network.feedforward(emily)
*_, frank_prediction = my_neural_network.feedforward(frank)
print(helpers.sigmoid(emily_prediction)) # 0.951 - F
print(helpers.sigmoid(frank_prediction)) # 0.039 - M

