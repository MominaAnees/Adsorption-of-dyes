"""
==================
1. ANN
==================
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor, MLPClassifier
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

data = pd.read_excel("Data.xlsx", skiprows=1)
X = data.drop(columns=['Stirringspeed', 'Temp', 'Time', 'Dosage', 'pH', 'Concentration'], axis=1)
y = data[['Concentration,Cf(mg/L)', 'Adsorption capacity(mg/g)', 'Adsorption efficiency(%)']]
# Convert column names to strings
X.columns = X.columns.astype(str)

# training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

# Hyperparameters for ANN
model = MLPRegressor(hidden_layer_sizes=(20, 20), activation='relu', solver='adam', max_iter=200, tol=1e-4, validation_fraction=0.15)
model.fit(X_train, y_train)

y_train = model.predict(X_train)
y_pred = model.predict(X_test)

# Mean Squared Error and R2
mse = mean_squared_error(y_test, y_pred)
print(y_pred)
print("Mean Squared Error:", mse)
print("R2:", r2_score(y_test, y_pred, sample_weight=None, force_finite=True))
