import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length), :]
        y = data[i + seq_length, 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64).reshape(-1, 1)


def sigmoid(x):
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(s):
    return s * (1.0 - s)


def dtanh(t):
    return 1.0 - t * t


class BaseModel:
    def __init__(self, input_size, hidden_size, lr, seed=42, clip_value=5.0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.clip_value = clip_value
        self.rng = np.random.default_rng(seed)

        self.Wy = self.rng.normal(0, 0.1, (hidden_size, 1))
        self.by = np.zeros((1,))

    def update(self, grads):
        for k, v in grads.items():
            v = np.clip(v, -self.clip_value, self.clip_value)
            setattr(self, k, getattr(self, k) - self.lr * v)

    def train_step(self, X, y):
        y_hat, cache = self.forward(X)
        loss = np.mean((y_hat - y) ** 2)
        dy = (2.0 / y.shape[0]) * (y_hat - y)
        grads = self.backward(cache, dy)
        self.update(grads)
        return float(loss)


class RNN(BaseModel):
    def __init__(self, input_size, hidden_size, lr, seed=42):
        super().__init__(input_size, hidden_size, lr, seed)
        self.Wxh = self.rng.normal(0, 0.1, (input_size, hidden_size))
        self.Whh = self.rng.normal(0, 0.1, (hidden_size, hidden_size))
        self.bh = np.zeros((hidden_size,))

    def forward(self, X):
        B, T, _ = X.shape
        h = np.zeros((B, self.hidden_size))
        hs = []
        for t in range(T):
            h = np.tanh(X[:, t, :] @ self.Wxh + h @ self.Whh + self.bh)
            hs.append(h)
        y_hat = h @ self.Wy + self.by
        return y_hat, {"X": X, "hs": hs}

    def backward(self, cache, dy):
        X, hs = cache["X"], cache["hs"]
        B, T, _ = X.shape

        dWy = hs[-1].T @ dy
        dby = dy.sum(axis=0)

        dh = dy @ self.Wy.T

        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)

        for t in reversed(range(T)):
            h_t = hs[t]
            h_prev = np.zeros((B, self.hidden_size)) if t == 0 else hs[t - 1]

            da = dh * dtanh(h_t)

            dWxh += X[:, t, :].T @ da
            dWhh += h_prev.T @ da
            dbh += da.sum(axis=0)

            dh = da @ self.Whh.T

        return {"Wxh": dWxh, "Whh": dWhh, "bh": dbh, "Wy": dWy, "by": dby}



class GRU(BaseModel):
    def __init__(self, input_size, hidden_size, lr, seed=42):
        super().__init__(input_size, hidden_size, lr, seed)
        self.Wxz = self.rng.normal(0, 0.1, (input_size, hidden_size))
        self.Whz = self.rng.normal(0, 0.1, (hidden_size, hidden_size))
        self.bz = np.zeros((hidden_size,))

        self.Wxr = self.rng.normal(0, 0.1, (input_size, hidden_size))
        self.Whr = self.rng.normal(0, 0.1, (hidden_size, hidden_size))
        self.br = np.zeros((hidden_size,))

        self.Wxh = self.rng.normal(0, 0.1, (input_size, hidden_size))
        self.Whh = self.rng.normal(0, 0.1, (hidden_size, hidden_size))
        self.bh = np.zeros((hidden_size,))

    def forward(self, X):
        B, T, _ = X.shape
        h = np.zeros((B, self.hidden_size))
        xs, zs, rs, hs, hts = [], [], [], [], []

        for t in range(T):
            x = X[:, t, :]

            z = sigmoid(x @ self.Wxz + h @ self.Whz + self.bz)
            r = sigmoid(x @ self.Wxr + h @ self.Whr + self.br)
            ht = np.tanh(x @ self.Wxh + (r * h) @ self.Whh + self.bh)

            h = (1 - z) * h + z * ht

            xs.append(x)
            zs.append(z)
            rs.append(r)
            hs.append(h)
            hts.append(ht)

        return h @ self.Wy + self.by, {
            "X": X, "xs": xs, "zs": zs, "rs": rs, "hs": hs, "hts": hts
        }

    def backward(self, cache, dy):
        X, xs, zs, rs, hs, hts = (
            cache["X"], cache["xs"], cache["zs"], cache["rs"], cache["hs"], cache["hts"]
        )

        B, T, _ = X.shape

        dWy = hs[-1].T @ dy
        dby = dy.sum(axis=0)

        dh = dy @ self.Wy.T

        dWxz = np.zeros_like(self.Wxz)
        dWhz = np.zeros_like(self.Whz)
        dbz = np.zeros_like(self.bz)

        dWxr = np.zeros_like(self.Wxr)
        dWhr = np.zeros_like(self.Whr)
        dbr = np.zeros_like(self.br)

        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)

        for t in reversed(range(T)):
            x, z, r, ht = xs[t], zs[t], rs[t], hts[t]
            h_prev = np.zeros((B, self.hidden_size)) if t == 0 else hs[t - 1]

            dz = dh * (ht - h_prev)
            dht = dh * z
            dh_prev = dh * (1 - z)

            da_h = dht * dtanh(ht)

            dWxh += x.T @ da_h
            dWhh += (r * h_prev).T @ da_h
            dbh += da_h.sum(axis=0)

            tmp = da_h @ self.Whh.T
            dr = tmp * h_prev
            dh_prev += tmp * r

            da_r = dr * dsigmoid(r)

            dWxr += x.T @ da_r
            dWhr += h_prev.T @ da_r
            dbr += da_r.sum(axis=0)

            dh_prev += da_r @ self.Whr.T

            da_z = dz * dsigmoid(z)

            dWxz += x.T @ da_z
            dWhz += h_prev.T @ da_z
            dbz += da_z.sum(axis=0)

            dh_prev += da_z @ self.Whz.T

            dh = dh_prev

        return {
            "Wxz": dWxz, "Whz": dWhz, "bz": dbz,
            "Wxr": dWxr, "Whr": dWhr, "br": dbr,
            "Wxh": dWxh, "Whh": dWhh, "bh": dbh,
            "Wy": dWy, "by": dby
        }

# LSTM unchanged (inherits clipping automatically)

class LSTM(BaseModel):
    def __init__(self, input_size, hidden_size, lr, seed=42):
        super().__init__(input_size, hidden_size, lr, seed)
        self.Wxi = self.rng.normal(0, 0.1, (input_size, hidden_size)); self.Whi = self.rng.normal(0, 0.1, (hidden_size, hidden_size)); self.bi = np.zeros((hidden_size,))
        self.Wxf = self.rng.normal(0, 0.1, (input_size, hidden_size)); self.Whf = self.rng.normal(0, 0.1, (hidden_size, hidden_size)); self.bf = np.zeros((hidden_size,))
        self.Wxo = self.rng.normal(0, 0.1, (input_size, hidden_size)); self.Who = self.rng.normal(0, 0.1, (hidden_size, hidden_size)); self.bo = np.zeros((hidden_size,))
        self.Wxc = self.rng.normal(0, 0.1, (input_size, hidden_size)); self.Whc = self.rng.normal(0, 0.1, (hidden_size, hidden_size)); self.bc = np.zeros((hidden_size,))

    def forward(self, X):
        B, T, _ = X.shape
        h = np.zeros((B, self.hidden_size)); c = np.zeros((B, self.hidden_size))
        xs, is_, fs_, os_, gs_, hs, cs = [], [], [], [], [], [], []
        for t in range(T):
            x = X[:, t, :]
            i = sigmoid(x @ self.Wxi + h @ self.Whi + self.bi)
            f = sigmoid(x @ self.Wxf + h @ self.Whf + self.bf)
            o = sigmoid(x @ self.Wxo + h @ self.Who + self.bo)
            g = np.tanh(x @ self.Wxc + h @ self.Whc + self.bc)
            c = f * c + i * g
            h = o * np.tanh(c)
            xs.append(x); is_.append(i); fs_.append(f); os_.append(o); gs_.append(g); hs.append(h); cs.append(c)
        return h @ self.Wy + self.by, {"X": X, "xs": xs, "is": is_, "fs": fs_, "os": os_, "gs": gs_, "hs": hs, "cs": cs}

    def backward(self, cache, dy):
        X, xs, is_, fs_, os_, gs_, hs, cs = cache["X"], cache["xs"], cache["is"], cache["fs"], cache["os"], cache["gs"], cache["hs"], cache["cs"]
        B, T, _ = X.shape
        dWy = hs[-1].T @ dy
        dby = dy.sum(axis=0)
        dh = dy @ self.Wy.T
        dc = np.zeros((B, self.hidden_size))
        dWxi = np.zeros_like(self.Wxi); dWhi = np.zeros_like(self.Whi); dbi = np.zeros_like(self.bi)
        dWxf = np.zeros_like(self.Wxf); dWhf = np.zeros_like(self.Whf); dbf = np.zeros_like(self.bf)
        dWxo = np.zeros_like(self.Wxo); dWho = np.zeros_like(self.Who); dbo = np.zeros_like(self.bo)
        dWxc = np.zeros_like(self.Wxc); dWhc = np.zeros_like(self.Whc); dbc = np.zeros_like(self.bc)
        for t in reversed(range(T)):
            x = xs[t]; i = is_[t]; f = fs_[t]; o = os_[t]; g = gs_[t]; c = cs[t]
            h_prev = np.zeros((B, self.hidden_size)) if t == 0 else hs[t - 1]
            c_prev = np.zeros((B, self.hidden_size)) if t == 0 else cs[t - 1]
            tanh_c = np.tanh(c)
            do = dh * tanh_c
            dc = dh * o * dtanh(tanh_c) + dc
            df = dc * c_prev
            di = dc * g
            dg = dc * i
            dc = dc * f
            da_o = do * dsigmoid(o)
            da_f = df * dsigmoid(f)
            da_i = di * dsigmoid(i)
            da_g = dg * dtanh(g)
            dWxo += x.T @ da_o; dWho += h_prev.T @ da_o; dbo += da_o.sum(axis=0)
            dWxf += x.T @ da_f; dWhf += h_prev.T @ da_f; dbf += da_f.sum(axis=0)
            dWxi += x.T @ da_i; dWhi += h_prev.T @ da_i; dbi += da_i.sum(axis=0)
            dWxc += x.T @ da_g; dWhc += h_prev.T @ da_g; dbc += da_g.sum(axis=0)
            dh = da_o @ self.Who.T + da_f @ self.Whf.T + da_i @ self.Whi.T + da_g @ self.Whc.T
        return {"Wxi": dWxi, "Whi": dWhi, "bi": dbi, "Wxf": dWxf, "Whf": dWhf, "bf": dbf, "Wxo": dWxo, "Who": dWho, "bo": dbo, "Wxc": dWxc, "Whc": dWhc, "bc": dbc, "Wy": dWy, "by": dby}

def batches(X, y, bs, shuffle=True, seed=42):
    idx = np.arange(len(X))
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for i in range(0, len(X), bs):
        j = idx[i:i + bs]
        yield X[j], y[j]


def evaluate(model, X, y, y_scaler, bs):
    preds = []
    trues = []
    for xb, yb in batches(X, y, bs, shuffle=False):
        p, _ = model.forward(xb)
        preds.append(p)
        trues.append(yb)

    p = y_scaler.inverse_transform(np.vstack(preds)).reshape(-1)
    t = y_scaler.inverse_transform(np.vstack(trues)).reshape(-1)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(t, p))),
        "mae": float(mean_absolute_error(t, p)),
        "r2": float(r2_score(t, p))
    }
    return metrics, p, t
    

def train(model, Xtr, ytr, Xva, yva, y_scaler, epochs, bs, seed):
    best = None
    best_loss = float("inf")
    for ep in range(1, epochs + 1):
        losses = []
        k = 0
        for xb, yb in batches(Xtr, ytr, bs, shuffle=True, seed=seed + ep):
            k+=1
            losses.append(model.train_step(xb, yb))
            #print(f"{k} batch losse {np.mean(losses)}") 
        #input()
        val,_,_ = evaluate(model, Xva, yva, y_scaler, bs)
        print(f"Epoch {ep:03d}/{epochs} | loss={np.mean(losses):.6f} | val_RMSE={val['rmse']:.6f}")
        if val["rmse"] < best_loss:
            best_loss = val["rmse"]
            best = {k: v.copy() for k, v in model.__dict__.items() if isinstance(v, np.ndarray)}
    if best is not None:
        for k, v in best.items():
            setattr(model, k, v)
    return model


def make_model(name, input_size, hidden_size, lr, seed):
    name = name.lower()
    if name == "rnn":
        return RNN(input_size, hidden_size, lr, seed)
    if name == "gru":
        return GRU(input_size, hidden_size, lr, seed)
    if name == "lstm":
        return LSTM(input_size, hidden_size, lr, seed)
    raise ValueError("model must be rnn, gru, lstm, or all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", default="lstm", choices=["rnn", "gru", "lstm", "all"])
    ap.add_argument("--seq-len", type=int, default=12)     
    ap.add_argument("--hidden-size", type=int, default=24)  

    ap.add_argument("--epochs", type=int, default=15)        
    ap.add_argument("--batch-size", type=int, default=16)     

    ap.add_argument("--learning-rate", type=float, default=0.01) 

    ap.add_argument("--seed", type=int, default=32)

    args = ap.parse_args()

    #df = pd.read_csv(args.csv).sort_index().reset_index(drop=True)
    df = pd.read_csv(args.csv)
    if "Usage_kWh" not in df.columns:
        raise ValueError("CSV must contain Usage_kWh")
    print("[CONFIG]")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    target_col = "Usage_kWh"
    print("[DATA]")
    print(f"Rows: {len(df)}")
    print(f"Target mean: {df[target_col].mean():.6f}")
    print(f"Target std:  {df[target_col].std():.6f}")

    n = len(df)
    n_train = max(1, int(n * 0.7))
    n_val = max(1, int(n * 0.15))
    if n_train + n_val >= n:
        raise ValueError("Bad split ratios")
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    drop_cols = [c for c in ["date"] if c in df.columns]
    cat_cols = [c for c in df.columns if c not in drop_cols + [target_col] and not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in df.columns if c not in drop_cols + [target_col] + cat_cols]

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)], remainder="drop")

    X_train = pre.fit_transform(train_df.drop(columns=[target_col] + drop_cols, errors="ignore"))
    X_val = pre.transform(val_df.drop(columns=[target_col] + drop_cols, errors="ignore"))
    X_test = pre.transform(test_df.drop(columns=[target_col] + drop_cols, errors="ignore"))

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(train_df[[target_col]]).reshape(-1)
    y_val = y_scaler.transform(val_df[[target_col]]).reshape(-1)
    y_test = y_scaler.transform(test_df[[target_col]]).reshape(-1)

    train_data = np.column_stack([y_train, X_train])
    val_data = np.column_stack([y_val, X_val])
    test_data = np.column_stack([y_test, X_test])

    Xtr, ytr = create_sequences(train_data, args.seq_len)
    Xva, yva = create_sequences(val_data, args.seq_len)
    Xte, yte = create_sequences(test_data, args.seq_len)

    print("[SHAPES]")
    print(f"Train seq: {Xtr.shape}")
    print(f"Val seq:   {Xva.shape}")
    print(f"Test seq:  {Xte.shape}")

    models = [args.model] if args.model != "all" else ["rnn", "gru", "lstm"]
    results = []
    plot_data = []
    for name in models:
        print(f"===== {name.upper()} =====")
        model = make_model(name, Xtr.shape[-1], args.hidden_size, args.learning_rate, args.seed)
        train(model, Xtr, ytr, Xva, yva, y_scaler, args.epochs, args.batch_size, args.seed)
        m, preds, trues = evaluate(model, Xte, yte, y_scaler, args.batch_size)
        
        print(f"Test | RMSE={m['rmse']:.6f} | MAE={m['mae']:.6f} | R2={m['r2']:.6f}")
        results.append({"model": name.upper(), **m})
        plot_data.append({
           "name": name.upper(),
           "preds": preds,
           "trues": trues
        })

    out = pd.DataFrame(results)[["model", "rmse", "mae", "r2"]]
    print("[COMPARISON]")
    print(out.to_string(index=False))
    best = out.sort_values("rmse").iloc[0]
    print(f"Best by RMSE: {best['model']} ({best['rmse']:.6f})")

    
    fig, axes = plt.subplots(len(plot_data), 1, figsize=(12, 5 * len(plot_data)))
    if len(plot_data) == 1: axes = [axes] # Если модель одна, делаем список для итерации

    for i, data in enumerate(plot_data):
        axes[i].plot(data['trues'], label='True', alpha=0.7)
        axes[i].plot(data['preds'], label='Pred', alpha=0.7)
        axes[i].set_title(f"Model: {data['name']}")
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
