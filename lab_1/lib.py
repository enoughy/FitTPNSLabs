import pandas as pd
import sklearn.tree as skt
import matplotlib.pyplot as plt
import seaborn as sns


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
df = pd.get_dummies(df, drop_first=True)

corr = df.corr()
y = df["NObeyesdad"]
df = df.drop("NObeyesdad", axis=1)


print(corr)
showHeatMap(corr)

# Удаляю признаки с большой кореляцией
df = df.drop("Gender_Male", axis=1)
df = df.drop("MTRANS_Public_Transportation", axis=1)
df = df.drop("family_history_with_overweight_yes", axis=1)

X = df

tree = skt.DecisionTreeClassifier(criterion="entropy")
tree.fit(df, y)
feature_importances = pd.Series(tree.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=False)  # по возрастанию

print(feature_importances)
