import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as mp
import math
import numpy


import numpy as np


def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))


def conditional_entropy_numeric(X_col, y, threshold):
    left_mask = X_col < threshold
    right_mask = X_col >= threshold

    y_left = y[left_mask]
    y_right = y[right_mask]

    if len(y_left) == 0 or len(y_right) == 0:
        return np.inf

    H_left = entropy(y_left)
    H_right = entropy(y_right)

    weighted_entropy = (len(y_left) / len(y)) * H_left + (
        len(y_right) / len(y)
    ) * H_right

    return weighted_entropy


def information_gain_numeric(X_col, y, threshold):
    return entropy(y) - conditional_entropy_numeric(X_col, y, threshold)


def best_split(X_col, y):
    sorted_values = np.sort(np.unique(X_col))

    thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2

    best_ig = -1
    best_threshold = None

    for t in thresholds:
        ig = information_gain_numeric(X_col, y, t)
        if ig > best_ig:
            best_ig = ig
            best_threshold = t

    return best_threshold, best_ig


def split_information_numeric(X_col, threshold):
    left = X_col < threshold
    right = X_col >= threshold

    counts = np.array([np.sum(left), np.sum(right)])
    probabilities = counts / len(X_col)

    return -np.sum(probabilities * np.log2(probabilities))


def gain_ratio_numeric(X_col, y):
    threshold, ig = best_split(X_col, y)
    si = split_information_numeric(X_col, threshold)

    if si == 0:
        return 0, threshold

    return ig / si, threshold


def main():
    df = pd.read_csv("data.csv")

    ordinal_mapping = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    ordinal_mapping_1 = {
        "Insufficient_Weight": 0,
        "Normal_Weight": 1,
        "Overweight_Level_I": 2,
        "Overweight_Level_II": 3,
        "Obesity_Type_I": 4,
        "Obesity_Type_II": 5,
        "Obesity_Type_III": 6,
    }

    df["CALC"] = df["CALC"].map(ordinal_mapping)
    df["CAEC"] = df["CAEC"].map(ordinal_mapping)

    df["NObeyesdad"] = df["NObeyesdad"].map(ordinal_mapping_1)
    y = df["NObeyesdad"]
    df = df.drop("NObeyesdad", axis=1)

    # Категориальные бинарные
    df = pd.get_dummies(df, drop_first=True)

    # Удaляю признаки с большой кареляцией
    df = df.drop("Gender_Male", axis=1)
    df = df.drop("MTRANS_Public_Transportation", axis=1)
    df = df.drop("family_history_with_overweight_yes", axis=1)

    gr_scores = {}
    for col in df.columns:
        gr, threshold = gain_ratio_numeric(df[col].values, y)
        gr_scores[col] = (gr, threshold)

    sorted_gr = sorted(gr_scores.items(), key=lambda x: x[1][0], reverse=True)

    print("Gain Ratio:")
    for feature, (score, threshold) in sorted_gr:
        print(f"{feature}: GR={score:.4f}")


if __name__ == "__main__":
    main()
