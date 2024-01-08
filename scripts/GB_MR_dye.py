"""
==================
7. GB for MR dye
==================
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_excel("Data.xlsx", skiprows=1)
data_subset = data.head(2000)
X = data_subset.drop(columns=['Stirringspeed', 'Temp', 'Time', 'Dosage', 'pH', 'Concentration'], axis=1)
y = data_subset[['Concentration,Cf(mg/L)', 'Adsorption capacity(mg/g)', 'Adsorption efficiency(%)']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=10000, learning_rate=0.1, max_depth=3, random_state=42)
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculate Mean Squared Error for each output
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

data = pd.read_excel("Data.xlsx", skiprows=1)
data_subset = data.head(1000)

X = data_subset.drop(columns=['Stirringspeed', 'Temp', 'Time', 'Dosage', 'pH', 'Concentration'], axis=1)
y = data_subset[['Adsorption capacity(mg/g)']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

gb_regressor = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42)
gb_regressor.fit(X_train, y_train)

train_losses = []
test_losses = []


for stage, y_train_pred in enumerate(gb_regressor.staged_predict(X_train)):
    train_loss = mean_squared_error(y_train, y_train_pred)
    train_losses.append(train_loss)

for stage, y_test_pred in enumerate(gb_regressor.staged_predict(X_test)):
    test_loss = mean_squared_error(y_test, y_test_pred)
    test_losses.append(test_loss)


train_losses = np.array(train_losses)
test_losses = np.array(test_losses)


plt.figure(figsize=(10, 6))
plt.plot(range(1, gb_regressor.n_estimators + 1), train_losses, label='Training Loss', linewidth=2)
plt.plot(range(1, gb_regressor.n_estimators + 1), test_losses, label='Testing Loss', linewidth=2)

plt.xlabel('Number of Estimators')
plt.ylabel('Loss')
plt.title('Training and Testing Loss vs Number of Estimators')
plt.legend()
plt.show()
