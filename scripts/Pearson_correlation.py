"""
==================
1. Pearson correlation
==================
"""
print("\033[14;1mMR DYE\033[m")
import pandas as pd
data = pd.read_excel("Data.xlsx", skiprows=1)
data.head()
# %%

import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_excel("Data.xlsx", skiprows=1)


#%%
#Correlation matrix for MR dye
# --------------
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='BrBG', fmt=".2f")
plt.title("Correlation Heatmap for MR dye")
plt.show()


#%%
#Correlation matrix for BG dye
# --------------
print("\033[14;1mBG DYE\033[m")
data = pd.read_excel("BG.xlsx")
print(data.describe())
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='BrBG', fmt=".2f")
plt.title("Correlation Heatmap for BG dye")
plt.show()

