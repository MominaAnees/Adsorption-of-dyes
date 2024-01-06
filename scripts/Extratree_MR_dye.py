"""
==================
5. Extra tree for MR dye
==================
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_excel("Data.xlsx", skiprows=1)
data_subset = data.head(2000)

X = data_subset[['Stirringspeed', 'Temp', 'Time', 'Dosage', 'pH', 'Concentration']]
y = data_subset[['Concentration,Cf(mg/L)', 'Adsorption capacity(mg/g)', 'Adsorption efficiency(%)']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
extra_trees_regressor = ExtraTreesRegressor(random_state=42)

# Hyperparameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# grid search
grid_search = GridSearchCV(extra_trees_regressor, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)
df = pd.read_excel('predicted_data_with_inputs_extrareg_MR.xlsx')
df_subset = df.head(200)
sns.set_theme(style='darkgrid')
# Create subplots in one line
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
