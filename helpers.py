import numpy as np

def sigmoid(total: int) -> int:
    return 1 / (1 + np.exp(-total))

def derivative_of_sigmoid(total: int) -> int:
    return np.exp(-total) / ((1 + np.exp(-total)) ** 2)

def mse(actuals: np.ndarray[int], predictions: np.ndarray[int]) -> int:
    squared_error = (actuals - predictions) ** 2
    mean = 0
    for error in squared_error:
        mean += error
    
    mean /= len(actuals)

    return mean
