import pandas as pd
from preprocess import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

TARGET_COLUMN = "default_payment_next_month"

df = pd.read_csv("cr.csv")
df = preprocessing(df, TARGET_COLUMN)
print("preprocessing done")

# --- X и y ---
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# def label_to_num(x):
#     if x <= 4:
#         return 0  # low
#     elif x < 6:
#         return 1  # medium
#     elif x < 7:
#         return 3  # high
#     else:
#         return 4
#

# y_train_num = y_train.apply(label_to_num)
# y_test_num = y_test.apply(label_to_num)
y_train_num = y_train
y_test_num = y_test

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = MLPClassifier(
    hidden_layer_sizes=(10),
    activation="logistic",  # relu
    solver="sgd",  # adam
    learning_rate_init=0.01,
    max_iter=500,
    random_state=4,
    verbose=True,
)

model.fit(X_train_scaled, y_train_num)

y_pred_class = model.predict(X_test_scaled)

# --- Accuracy ---
accuracy = accuracy_score(y_test_num, y_pred_class)
print("Accuracy:", accuracy)

# for true, pred in zip(y_test_num, y_pred_class):
#     print(f"true: {true}, pred: {pred}, ok: {true == pred}")
# --- loss ---
plt.plot(model.loss_curve_)
plt.xlabel("Итерация")
plt.ylabel("Loss")
plt.title("График функции потерь MLPClassifier")
plt.show()
