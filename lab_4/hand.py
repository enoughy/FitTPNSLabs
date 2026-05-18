import numpy as np
import gzip
import struct
import urllib.request
import os
import tensorflow as tf

def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(np.float32)


def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)


def cross_entropy(pred, label):
    return -np.log(pred[label] + 1e-8)


# CONV2D


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weights = (
            np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1
        )

        self.biases = np.zeros(out_channels)

    def forward(self, x):

        self.input = x

        c, h, w = x.shape
        k = self.kernel_size

        out_h = h - k + 1
        out_w = w - k + 1

        output = np.zeros((self.out_channels, out_h, out_w))

        for oc in range(self.out_channels):
            for ic in range(self.in_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        region = x[ic, i : i + k, j : j + k]

                        output[oc, i, j] += np.sum(region * self.weights[oc, ic])

            output[oc] += self.biases[oc]

        self.output = output
        return output

    def backward(self, grad_output, lr):

        c, h, w = self.input.shape
        k = self.kernel_size

        out_h = h - k + 1
        out_w = w - k + 1

        grad_input = np.zeros_like(self.input)

        grad_weights = np.zeros_like(self.weights)
        grad_biases = np.zeros_like(self.biases)

       
        for oc in range(self.out_channels):
            grad_biases[oc] = np.sum(grad_output[oc])

            for ic in range(self.in_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        region = self.input[ic, i : i + k, j : j + k]

                        grad_weights[oc, ic] += region * grad_output[oc, i, j]

                        grad_input[ic, i : i + k, j : j + k] += (
                            self.weights[oc, ic] * grad_output[oc, i, j]
                        )

      
        self.weights -= lr * grad_weights
        self.biases -= lr * grad_biases

        return grad_input



class AvgPool2D:
    def __init__(self, size=2):
        self.size = size

    def forward(self, x):

        self.input = x

        c, h, w = x.shape
        s = self.size

        out_h = h // s
        out_w = w // s

        output = np.zeros((c, out_h, out_w))

        for ch in range(c):
            for i in range(out_h):
                for j in range(out_w):
                    region = x[ch, i * s : (i + 1) * s, j * s : (j + 1) * s]

                    output[ch, i, j] = np.mean(region)

        return output

    def backward(self, grad_output):

        c, out_h, out_w = grad_output.shape
        s = self.size

        grad_input = np.zeros_like(self.input)

        for ch in range(c):
            for i in range(out_h):
                for j in range(out_w):
                    grad = grad_output[ch, i, j] / (s * s)

                    grad_input[ch, i * s : (i + 1) * s, j * s : (j + 1) * s] += grad

        return grad_input



class FullyConnected:
    def __init__(self, in_features, out_features):

        self.weights = np.random.randn(in_features, out_features) * 0.1

        self.biases = np.zeros(out_features)

    def forward(self, x):

        self.input = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, grad_output, lr):

        grad_weights = np.outer(self.input, grad_output)

        grad_biases = grad_output

        grad_input = np.dot(self.weights, grad_output)

        self.weights -= lr * grad_weights
        self.biases -= lr * grad_biases

        return grad_input



class LeNet5:
    def __init__(self):

        self.conv1 = Conv2D(1, 6, 5)
        self.pool1 = AvgPool2D(2)

        self.conv2 = Conv2D(6, 16, 5)
        self.pool2 = AvgPool2D(2)

        self.fc1 = FullyConnected(16 * 4 * 4, 120)

        self.fc2 = FullyConnected(120, 84)

        self.fc3 = FullyConnected(84, 10)

    def forward(self, x):

        # CONV1
        self.x1 = self.conv1.forward(x)
        self.a1 = relu(self.x1)

        # POOL1
        self.p1 = self.pool1.forward(self.a1)

        # CONV2
        self.x2 = self.conv2.forward(self.p1)
        self.a2 = relu(self.x2)

        # POOL2
        self.p2 = self.pool2.forward(self.a2)

        # FLATTEN
        self.flat = self.p2.reshape(-1)

        # FC1
        self.f1 = self.fc1.forward(self.flat)
        self.a3 = relu(self.f1)

        # FC2
        self.f2 = self.fc2.forward(self.a3)
        self.a4 = relu(self.f2)

        # FC3
        self.f3 = self.fc3.forward(self.a4)

        return softmax(self.f3)

    def backward(self, pred, label, lr):


        grad = pred.copy()
        grad[label] -= 1

        # FC3

        grad = self.fc3.backward(grad, lr)

        # FC2

        grad *= relu_derivative(self.f2)

        grad = self.fc2.backward(grad, lr)

        # FC1

        grad *= relu_derivative(self.f1)

        grad = self.fc1.backward(grad, lr)


        grad = grad.reshape(self.p2.shape)

        # POOL2

        grad = self.pool2.backward(grad)

        # CONV2

        grad *= relu_derivative(self.x2)

        grad = self.conv2.backward(grad, lr)

        # POOL1

        grad = self.pool1.backward(grad)

        # CONV1

        grad *= relu_derivative(self.x1)

        self.conv1.backward(grad, lr)


# ACCURACY


def accuracy(model, images, labels, samples=1000):

    correct = 0

    for i in range(samples):
        x = images[i].reshape(1, 28, 28)

        pred = model.forward(x)

        if np.argmax(pred) == labels[i]:
            correct += 1

    return correct / samples * 100


# TRAIN


def train(
    model, train_images, train_labels, test_images, test_labels, epochs=3, lr=0.001
):

    for epoch in range(epochs):
        total_loss = 0
        batch_loss = 0

        for i in range(len(train_images)):
            x = train_images[i].reshape(1, 28, 28)

            y = train_labels[i]

            pred = model.forward(x)

            loss = cross_entropy(pred, y)

            batch_loss += loss

            total_loss += loss

            model.backward(pred, y, lr)

            if i % 500 == 0:
                avg_batch_loss = batch_loss/500 if i!=0 else batch_loss
                print(f"Epoch {epoch + 1} | Sample {i} | Loss: {avg_batch_loss:.4f}")
                batch_loss = 0;

        avg_loss = total_loss / len(train_images)

        acc = accuracy(model, test_images, test_labels)

        print("\n====================")
        print(f"Epoch {epoch + 1} completed")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Accuracy: {acc:.2f}%")
        print("====================\n")

    # =========================================================
    # MAIN
    # =========================================================


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

model = LeNet5()
train(model, x_train[:5000], y_train[:5000], x_test, y_test, epochs=3, lr=0.001)

final_acc = accuracy(model, x_test, y_test)
print(f"Final Accuracy: {final_acc:.2f}%")
