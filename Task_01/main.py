# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the iris dataset from seaborn
iris = sns.load_dataset("iris")

# Data Inspection

print("Shape of the dataset:", iris.shape)


print("Column names:", iris.columns.tolist())


print("\nFirst five rows:")
print(iris.head())


print("\nDataset Info:")
print(iris.info())


print("\nSummary Statistics:")
print(iris.describe())



sns.set(style="whitegrid")


sns.pairplot(iris, hue="species", height=2.5)
plt.suptitle("Scatter Plot Matrix", y=1.02)
plt.show()


iris.hist(edgecolor='black', figsize=(10, 8))
plt.suptitle("Histograms of Each Feature")
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
for idx, column in enumerate(iris.columns[:-1]):
    plt.subplot(1, 4, idx+1)
    sns.boxplot(y=iris[column])
    plt.title(column)
plt.suptitle("Box Plots for Feature Distribution")
plt.tight_layout()
plt.show()
