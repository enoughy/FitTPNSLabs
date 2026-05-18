import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd


def conf_matrix(y_test, predicted_labels):
    cm = confusion_matrix(y_test, predicted_labels)
    df = pd.DataFrame(
        cm,
        index=[f"true {i}" for i in range(10)],
        columns=[f"pred {i}" for i in range(10)],
    )
    print(df)


def show_wrongs(model):
    predictions = model.predict(x_test)

    predicted_labels = np.argmax(predictions, axis=1)

    wrong_indices = np.where(predicted_labels != y_test)[0]
    conf_matrix(y_test, predicted_labels)

    plt.figure(figsize=(12, 8))

    for i in range(9):
        idx = wrong_indices[i]

        plt.subplot(3, 3, i + 1)

        plt.imshow(x_test[idx].squeeze(), cmap="gray")

        plt.title(f"True: {y_test[idx]}\nPred: {predicted_labels[idx]}")

        plt.axis("off")

    plt.tight_layout()
    plt.show()


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# Нормализация
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

model = models.Sequential()

# C1: Conv
model.add(layers.Conv2D(6, (5, 5), activation="tanh", input_shape=(28, 28, 1)))

# S2: Avg Pooling
model.add(layers.AveragePooling2D(pool_size=(2, 2)))

# C3: Conv
model.add(layers.Conv2D(16, (5, 5), activation="tanh"))

# S4: Avg Pooling
model.add(layers.AveragePooling2D(pool_size=(2, 2)))

# Flatten
model.add(layers.Flatten())

# C5: Fully Connected
model.add(layers.Dense(120, activation="tanh"))

# F6
model.add(layers.Dense(84, activation="tanh"))

# Output
model.add(layers.Dense(10, activation="softmax"))

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

show_wrongs(model)

plt.plot(history.history["accuracy"], label="train")
plt.legend()
plt.title("Accuracy")
plt.show()

plt.plot(history.history["loss"], label="train loss")
plt.legend()
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
