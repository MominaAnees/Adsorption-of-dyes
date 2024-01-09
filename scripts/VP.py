"""
====================
9. Violin plots 
====================
"""

#%%
# MR dye
# --------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb


data = pd.read_excel("Data.xlsx", skiprows=1)
data_subset = data.head(2000)
X = data_subset.drop(columns=['Stirringspeed', 'Temp', 'Time', 'Dosage', 'pH', 'Concentration'], axis=1)
y = data_subset[[ 'Adsorption capacity(mg/g)']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)


models = {
    'Random Forest': RandomForestRegressor(n_estimators=200, max_features='sqrt', bootstrap=True, max_depth=None, oob_score=True, random_state=42),
    'Extra Trees': ExtraTreesRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'SVM RBF': SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1),
    'SVM Poly': SVR(kernel='poly', C=1.0, gamma='scale', epsilon=0.1),
    'XGBoost': xgb.XGBRegressor(n_estimators=200, min_child_weight=3, max_depth=7, learning_rate=0.1, colsample_bytree=0.8),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(20, 20), activation='relu', solver='adam', max_iter=500, tol=1e-4, 
                     validation_fraction=0.15),
}


predictions = {}
for name, model in models.items():
    if not hasattr(model, 'oob_score_') or not model.oob_score_:
        model.fit(X_train, y_train)
    predictions[name] = model.predict(X_test)


df_predictions = pd.DataFrame(predictions)

# Plotting
plt.figure(figsize=(14, 8))
sns.violinplot(data=df_predictions, palette='viridis')
plt.title('Comparison of Output Distributions for Different Models')
plt.ylabel('MR Adsorption capacity distribution', fontsize=16)
plt.xlabel('ML Models', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.show()

#%%
# BG dye
# --------------

data = pd.read_excel("BG.xlsx")
data_subset = data.head(2000)
X = data_subset.drop(columns=['Stirringspeed', 'Temp', 'Time', 'Dosage', 'pH', 'Concentration'], axis=1)
y = data_subset[[ 'Adsorption capacity(mg/g)']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

models = {
    'Random Forest': RandomForestRegressor(n_estimators=200, max_features='sqrt', bootstrap=True, max_depth=None, oob_score=True, random_state=42),
    'Extra Trees': ExtraTreesRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'SVM RBF': SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1),
    'SVM Poly': SVR(kernel='poly', C=1.0, gamma='scale', epsilon=0.1),
    'XGBoost': xgb.XGBRegressor(n_estimators=200, min_child_weight=3, max_depth=7, learning_rate=0.1, colsample_bytree=0.8),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(20, 20), activation='relu', solver='adam', max_iter=500, tol=1e-4, 
                     validation_fraction=0.15),
}


predictions = {}
for name, model in models.items():
    if not hasattr(model, 'oob_score_') or not model.oob_score_:
        model.fit(X_train, y_train)
    predictions[name] = model.predict(X_test)


df_predictions = pd.DataFrame(predictions)

# Plotting
plt.figure(figsize=(14, 8))
sns.violinplot(data=df_predictions, palette='viridis')
plt.title('Comparison of Output Distributions for Different Models')
plt.ylabel('BG adsorption capacity distribution', fontsize=16)
plt.xlabel('ML Models', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.show()
