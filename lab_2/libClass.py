import pandas as pd
from preprocess import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

TARGET_COLUMN = "Price"

df = pd.read_csv("Laptop_price.csv")
df = preprocessing(df, TARGET_COLUMN, "REG")

# --- X и y ---
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- масштабирование ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))

model = MLPRegressor(
    hidden_layer_sizes=(10,),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42,
    verbose=True,
)

model.fit(X_train_scaled, y_train_scaled)

# --- предсказание ---
y_pred = model.predict(X_test_scaled)

# --- Accuracy ---
r2 = r2_score(y_test_scaled, y_pred)
print("R2 score:", r2)

# --- loss ---
plt.plot(model.loss_curve_)
plt.xlabel("Эпоха")
plt.ylabel("Loss")
plt.title("График функции потерь ")
plt.show()
