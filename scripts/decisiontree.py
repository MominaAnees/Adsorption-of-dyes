"""
==================
7. Decision trees
==================
"""
#%%
# MR dye
## --------------

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_excel("Data.xlsx", skiprows=1)
data_subset = data.head(2000)

X = data_subset[['Stirringspeed', 'Temp', 'Time', 'Dosage', 'pH', 'Concentration']]
y = data_subset[['Concentration,Cf(mg/L)', 'Adsorption capacity(mg/g)', 'Adsorption efficiency(%)']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

#%%
df = pd.read_excel('predicted_data_with_inputs_decisiontree_MR.xlsx')
df_subset = df.head(200)
df.head(200)

sns.regplot(x='Concentration,Cf(mg/L)', y ='Predicted_Concentration,Cf(mg/L)', data = df_subset,
           scatter_kws={"s": 60, "color": 'snow', "marker": 'o', "edgecolor": 'blue'},
            line_kws={"color": 'indigo', "linestyle": '--'})
plt.tight_layout()
plt.show()

#%%
sns.regplot(x='Adsorption capacity(mg/g)', y ='Predicted_Adsorption capacity(mg/g)', data = df_subset,
            scatter_kws={"s": 60, "color": 'snow', "marker": 'o', "edgecolor": 'blue'},
            line_kws={"color": 'indigo', "linestyle": '--'})
plt.tight_layout()
plt.show()

#%%
sns.regplot(x='Adsorption efficiency(%)', y ='Predicted_Adsorption efficiency(%)', data = df_subset,
            scatter_kws={"s": 60, "color": 'snow', "marker": 'o', "edgecolor": 'blue'},
          line_kws={"color": 'indigo', "linestyle": '--'})
plt.tight_layout()
plt.show()


#%%
# BG dye
## --------------
data = pd.read_excel("BG.xlsx")
data_subset = data.head(2000)

X = data_subset[['Stirringspeed', 'Temp', 'Time', 'Dosage', 'pH', 'Concentration']]
y = data_subset[['Concentration,Cf(mg/L)', 'Adsorption capacity(mg/g)', 'Adsorption efficiency(%)']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

#%%
df = pd.read_excel('predicted_data_with_inputs_decisiontree_BG.xlsx')
df_subset = df.head(200)
df.head(200)
df_subset = df.head(200)
sns.set_theme(style='darkgrid')
# Create subplots in one line
sns.regplot(x='Concentration,Cf(mg/L)', y ='Predicted_Concentration,Cf(mg/L)', data = df_subset,
           scatter_kws={"s": 60, "color": 'snow', "marker": 'o', "edgecolor": 'blue'},
            line_kws={"color": 'indigo', "linestyle": '--'})
plt.tight_layout()
plt.show()
#%%
sns.regplot(x='Adsorption capacity(mg/g)', y ='Predicted_Adsorption capacity(mg/g)', data = df_subset,
            scatter_kws={"s": 60, "color": 'snow', "marker": 'o', "edgecolor": 'blue'},
            line_kws={"color": 'indigo', "linestyle": '--'})
plt.tight_layout()
plt.show()

#%%
sns.regplot(x='Adsorption efficiency(%)', y ='Predicted_Adsorption efficiency(%)', data = df_subset,
            scatter_kws={"s": 60, "color": 'snow', "marker": 'o', "edgecolor": 'blue'},
          line_kws={"color": 'indigo', "linestyle": '--'})
plt.tight_layout()
plt.show()
