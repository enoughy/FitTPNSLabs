import pandas as pd
import sklearn.tree as skt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

np.seterr(over="ignore")


def showHeatMap(corr):
    plt.figure(figsize=(30, 12))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        cbar=True,
        linewidths=0.5,
        linecolor="gray",
    )
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.title("Correlation Heatmap", fontsize=18)
    plt.tight_layout()
    plt.show()


def preprocessing(df, target, mode="CLASS"):
    df = pd.get_dummies(df, drop_first=True)
    df_c = df.copy()
    y = df_c[target]
    df_c = df_c.drop(target, axis=1)
    corr = df_c.corr().abs()

    # showHeatMap(corr)

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]

    print("Удаляем:", to_drop)

    df_c = df_c.drop(columns=to_drop)

    if mode == "CLASS":
        tree = skt.DecisionTreeClassifier(criterion="entropy")
    elif mode == "REG":
        tree = skt.DecisionTreeRegressor(criterion="absolute_error")
    else:
        print("Incorect method")
        return

    tree.fit(df_c, y)
    feature_importances = pd.Series(tree.feature_importances_, index=df_c.columns)
    feature_importances = feature_importances.sort_values(
        ascending=False
    )  # по возрастанию

    print(feature_importances)
    return df
