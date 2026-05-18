from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as mp
import seaborn
import random
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from preprocess import preprocessing

from sklearn.metrics import accuracy_score

TARGET_COLUMN = "quality"
#TARGET_COLUMN = "default_payment_next_month"

class Neuron:
    output = None
    inputs = None
    weights = None
    delta = 0
    bias = None

    def __init__(self, inputs_n):
        rng = np.random.default_rng()

        self.weights = rng.random(inputs_n) * 0.01
        self.bias = rng.random()

    def activate(self, inputs):
        self.inputs = inputs
        self.output = self.logistic_function(np.dot(self.weights, inputs) + self.bias)
        return self.output

    def add_delta(self, delta):
        self.delta += delta

    def get_delta(self):
        return self.delta

    def set_delta_zero(self):
        self.delta = 0

    def get_delta_next(self):
        return self.delta * np.array(self.weights)

    def recalculating_weight(self, n):
        self.weights = (
            self.weights
            + n
            * self.delta
            * self.derivative_of_logistic_function(self.output)
            * np.array(self.inputs)
        )

        self.bias = self.bias + n * self.delta * self.derivative_of_logistic_function(
            self.output
        )

    def recalculating_weight_output(self, n):
        self.weights = (
            self.weights
            + n
            * self.delta
            * self.derivative_of_logistic_function(self.output)
            * np.array(self.inputs)
        )
        self.bias += n * self.delta * self.derivative_of_logistic_function(self.output)

    def logistic_function(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative_of_logistic_function(self, x):
        return x * (1 - x)


class MultyLayerPerceptron:
    hidden_layers = None
    output_layer = None
    output_layer_size = None
    iters = None
    n = None

    def __init__(self, iters, input_layer_size, output_layer_size, n, layers_sizes=[]):
        self.hidden_layers = [[]]
        self.hidden_layers[0] = [
            Neuron(input_layer_size) for _ in range(layers_sizes[0])
        ]
        for i in range(1, len(layers_sizes)):
            self.hidden_layers.append(
                [Neuron(layers_sizes[i - 1]) for _ in range(layers_sizes[i])]
            )
        self.output_layer = [Neuron(layers_sizes[-1]) for _ in range(output_layer_size)]
        self.output_layer_size = output_layer_size
        self.iters = iters
        self.n = n

    def softmax(self, x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def predict(self, input_layer):
        layer_outputs = [
            neuron.activate(input_layer) for neuron in self.hidden_layers[0]
        ]
        for i in range(1, len(self.hidden_layers)):
            layer_outputs = [
                neuron.activate(layer_outputs) for neuron in self.hidden_layers[i]
            ]

        outputs = [neuron.activate(layer_outputs) for neuron in self.output_layer]
        probs = self.softmax(outputs)

        return np.argmax(probs)

    # def back_propagation(self, inputs, correct_output, nu):
    #     output, probs, target = self._forward_and_prepare(inputs, correct_output)
    #
    #     self._compute_output_deltas(probs, target)
    #     self._compute_last_hidden_deltas()
    #     self._compute_hidden_deltas()
    #
    #     self._update_hidden_weights(nu)
    #     self._update_output_weights(nu)
    def calculate_weights(self):
        for neuron in self.output_layer:
            neuron.recalculating_weight_output(self.n)
        for layer in self.hidden_layers:
            for neuron in layer:
                neuron.recalculating_weight(self.n)

    def spread_delta_layer(self, neurons, delta):
        for delta_from_one in delta:
            for j in range(len(neurons)):
                # print(delta)
                # print(delta_from_one)
                # input()
                neurons[j].add_delta(delta_from_one[j])
            # print(delta)
            # input()
        return [neuron.get_delta_next() for neuron in neurons]

    def spread_delta_all(self, delta):
        next_delta = self.spread_delta_layer(self.output_layer, delta)
        for layer in reversed(self.hidden_layers):
            next_delta = self.spread_delta_layer(layer, next_delta)

    def set_delta_zero(self):
        for neuron in self.output_layer:
            neuron.set_delta_zero()
        for layer in self.hidden_layers:
            for neuron in layer:
                neuron.set_delta_zero()

    def forward(self, inputs, correct_output):
        output = self.predict(inputs)

        outputs = [neuron.output for neuron in self.output_layer]
        probs = self.softmax(outputs)

        return output, probs

    def study(self, inputs, correct_output):
        output, probs = self.forward(inputs, correct_output)
        target = np.zeros(self.output_layer_size)
        error = [[0 for _ in range(self.output_layer_size)] for _ in range(1)]

        target[correct_output] = 1

        for i in range(len(probs)):
            error[0][i] = target[i] - probs[i]

        # back preprocessing
        self.spread_delta_all(error)
        # get input layer delta for transfer to next layer
        input_layer_delta = [neuron.get_delta for neuron in self.hidden_layers[0]]
        self.calculate_weights()
        self.set_delta_zero()
        return input_layer_delta

    def fit(self, inputs, outputs):
        for j in range(self.iters):
            for i in range(len(inputs)):
                self.study(inputs[i], outputs.iloc[i])
            print("iters: ", j)


def main():
#    df = pd.read_csv("cr.csv")
    df = pd.read_csv("wine.csv")

    df = preprocessing(df, TARGET_COLUMN)

    # Разделяем на X и y
    y = df[TARGET_COLUMN]
    X = df.drop(TARGET_COLUMN, axis=1)
    print(y)

    # Делим на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    y_train = y_train.apply(
        lambda x: 0 if x <= 4 else (1 if (x >= 5 and x <= 6) else 2)
    )
    y_test = y_test.apply(lambda x: 0 if x <= 4 else (1 if (x >= 5 and x <= 6) else 2))

    # y_train = y_train.apply(
    #     lambda x: 0 if x <= 4 else (1 if x < 5 else (2 if x < 6 else 3))
    # )
    #
    # y_test = y_test.apply(
    #     lambda x: 0 if x <= 4 else (1 if x < 5 else (2 if x < 6 else 3))
    # )

    # Масштабируем
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    df_pred_res = pd.DataFrame(
        {"Actual": y_test, }
    )
    # print(df_pred_res.to_string())
    mlp_class = MultyLayerPerceptron(
        3,
        X_train_scaled.shape[1],
        3,
        0.05,
        [100],
    )
    mlp_class.fit(X_train_scaled, y_train)
    y_pred = []

    for i in X_test_scaled:
        y_pred.append(mlp_class.predict(i))
    df_pred_res = pd.DataFrame(
        {"Actual": y_test, "Predicted": y_pred, "Diff": y_test - y_pred}
    )
    print(df_pred_res.to_string())
    # print(100 - sum(abs(df_pred_res["Diff"])) / len(df_pred_res["Diff"]) * 100, "%")
    print("Accuracy:", accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    main()
