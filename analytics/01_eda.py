import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = sns.load_dataset("titanic")
df.to_csv("titanic.csv", index=False)

print(df.info())
print(df.describe())
print(df.shape)

missing = df.isnull().mean() * 100
print(missing[missing > 0])

df = df.dropna(subset=["age"])  # <5% missing
df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)  # impute

sns.histplot(df["age"], kde=True); plt.show()
sns.boxplot(x=df["age"]); plt.show()

sns.histplot(df["fare"], kde=True); plt.show()
sns.boxplot(x=df["fare"]); plt.show()

Q1, Q3 = df["fare"].quantile([0.25, 0.75])
IQR = Q3 - Q1
outliers_fare = ((df["fare"] < Q1 - 1.5*IQR) | (df["fare"] > Q3 + 1.5*IQR)).sum()
print("Fare outliers:", outliers_fare)

print(df.groupby("sex")["survived"].mean())
print(df.groupby("pclass")["survived"].mean())
print(df.groupby(["sex","pclass"])["survived"].mean())

corr = df[["survived","pclass","age","sibsp","parch","fare"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm"); plt.show()

sns.barplot(x="sex", y="survived", data=df); plt.show()
sns.barplot(x="pclass", y="survived", data=df); plt.show()
sns.scatterplot(x="age", y="fare", hue="survived", data=df); plt.show()
sns.pairplot(df[["survived","pclass","age","fare"]], hue="survived"); plt.show()

scaler = StandardScaler()
df[["age","fare"]] = scaler.fit_transform(df[["age","fare"]])
print(df[["age","fare"]].describe())