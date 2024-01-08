"""
==================
8. GB for BG dye
==================
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_excel("BG.xlsx")
data_subset = data.head(2000)
X = data_subset.drop(columns=['Stirringspeed', 'Temp', 'Time', 'Dosage', 'pH', 'Concentration'], axis=1)
y = data_subset[['Concentration,Cf(mg/L)', 'Adsorption capacity(mg/g)', 'Adsorption efficiency(%)']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
gb_regressor = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
gb_regressor.fit(X_train, y_train.values.ravel())  

test_score = np.zeros((gb_regressor.n_estimators,))
for i, y_pred in enumerate(gb_regressor.staged_predict(X_test)):
    test_score[i] = mean_squared_error(y_test, y_pred)

# Plotting
fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title("Deviance")
plt.plot(
    np.arange(gb_regressor.n_estimators) + 1,
    gb_regressor.train_score_,
    "b-",
    label="Training Set Deviance",
)
plt.plot(
    np.arange(gb_regressor.n_estimators) + 1, test_score, "r-", label="Test Set Deviance"
)
plt.legend(loc="upper right")
plt.xlabel("n_estimator")
plt.ylabel("Deviance")
fig.tight_layout()
plt.show()
gb_regressor.fit(X_train, y_train)


y_pred = gb_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)
