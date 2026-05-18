import pandas as pd

# читаем файлы
df1 = pd.read_csv("winequality-white.csv", sep=";")
df2 = pd.read_csv("winequality-red.csv", sep=";")

# добавляем колонку-источник
df1["white"] = 1
df2["white"] = 0

# объединяем
df = pd.concat([df1, df2], ignore_index=True)

# сохраняем (если нужно)
df.to_csv("wine.csv", index=False)
