"""
==================
11. SVMpoly for MR dye
==================
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel("Data.xlsx", skiprows=1)


data_subset = data.head(2000)
X = data_subset.drop(columns=['Stirringspeed', 'Temp', 'Time', 'Dosage', 'pH', 'Concentration'], axis=1)
y = data_subset[['Concentration,Cf(mg/L)', 'Adsorption capacity(mg/g)', 'Adsorption efficiency(%)']]
X.columns = X.columns.astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

svr = SVR(kernel='poly', C=1.0, gamma='scale', epsilon=0.1)
model = MultiOutputRegressor(svr)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
print("Mean Squared Error for Adsorption capacity(mg/g):", mse[1])
print("R2 Score for Adsorption capacity(mg/g):", r2[1])
df = pd.read_excel('predicted_data_with_inputs_SVRpoly_MR.xlsx')
df_subset = df.head(200)
df.head(200)
df_subset = df.head(200)
sns.set_theme(style='darkgrid')
# Subplots
sns.regplot(x='Concentration,Cf(mg/L)', y ='Predicted_Concentration,Cf(mg/L)', data = df_subset,
           scatter_kws={"s": 60, "color": 'snow', "marker": 'o', "edgecolor": 'blue'},
            line_kws={"color": 'indigo', "linestyle": '--'})
plt.tight_layout()
plt.show()
sns.regplot(x='Adsorption capacity(mg/g)', y ='Predicted_Adsorption capacity(mg/g)', data = df_subset,
            scatter_kws={"s": 60, "color": 'snow', "marker": 'o', "edgecolor": 'blue'},
            line_kws={"color": 'indigo', "linestyle": '--'})
plt.tight_layout()
plt.show()
sns.regplot(x='Adsorption efficiency(%)', y ='Predicted_Adsorption efficiency(%)', data = df_subset,
            scatter_kws={"s": 60, "color": 'snow', "marker": 'o', "edgecolor": 'blue'},
          line_kws={"color": 'indigo', "linestyle": '--'})
plt.tight_layout()
plt.show()
