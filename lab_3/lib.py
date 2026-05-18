import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score


def prepare_data(file_path):
    df = pd.read_csv(file_path)

    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        if col == df.columns[0]:
            continue
        df[col] = le.fit_transform(df[col])

    target_col = 'Usage_kWh'

    feature_cols = [c for c in df.columns if c != df.columns[0]]

    feature_cols = [target_col] + [c for c in feature_cols if c != target_col]

    data = df[feature_cols].values.astype(float)

    return data, feature_cols


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length), :]
        y = data[i + seq_length, 0]   
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


class PowerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_dim, model_type='RNN'):
        super(PowerModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.model_type = model_type

        if model_type == 'RNN':
            self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True)
        elif model_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        elif model_type == 'GRU':
            self.rnn = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

        if self.model_type == 'LSTM':
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)

        return self.fc(out[:, -1, :])


file_path = 'Steel_industry_data.csv'
seq_length = 24
epochs = 5
batch_size = 32

data, cols = prepare_data(file_path)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

X, y = create_sequences(data_scaled, seq_length)

split = int(0.8 * len(X))

X_train = torch.FloatTensor(X[:split])
X_test = torch.FloatTensor(X[split:])
y_train = torch.FloatTensor(y[:split])
y_test = torch.FloatTensor(y[split:])

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=batch_size,
    shuffle=False
)

architectures = ['RNN', 'LSTM', 'GRU']
results = {}

plt.figure(figsize=(18, 5))

target_index = 0

for i, arch in enumerate(architectures):
    print(f"Обработка {arch}...")

    model = PowerModel(
        input_dim=X.shape[2],
        hidden_dim=64,
        output_dim=1,
        layer_dim=2,
        model_type=arch
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()

            pred = model(xb)
            loss = criterion(pred, yb.unsqueeze(1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss/len(train_loader):.6f}")

    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test).cpu().numpy()
        actual_scaled = y_test.cpu().numpy().reshape(-1, 1)

    t_min = scaler.data_min_[target_index]
    t_max = scaler.data_max_[target_index]

    preds = preds_scaled * (t_max - t_min) + t_min
    actual = actual_scaled * (t_max - t_min) + t_min

    mse = mean_squared_error(actual, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, preds)

    results[arch] = [mse, rmse, r2]

    plt.subplot(1, 3, i + 1)
    plt.plot(actual[:100], label='Real', alpha=0.6)
    plt.plot(preds[:100], label='Pred')
    plt.title(f'{arch}\nRMSE: {rmse:.4f}')
    plt.legend()

plt.tight_layout()
plt.show()

print("\nСравнение результатов:")
for arch, (mse, rmse, r2) in results.items():
    print(f"{arch}: MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
