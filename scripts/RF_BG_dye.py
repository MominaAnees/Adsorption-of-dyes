User
"""
==================
5. RF for BG dye
==================
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
data = pd.read_excel("BG.xlsx")
X = data.drop(columns=['Stirringspeed', 'Temp', 'Time', 'Dosage', 'pH', 'Concentration'], axis=1)
y = data[['Concentration,Cf(mg/L)', 'Adsorption capacity(mg/g)', 'Adsorption efficiency(%)']]

X.columns = X.columns.astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
model = RandomForestRegressor(n_estimators=200, max_features='sqrt',bootstrap=True, max_depth=None, oob_score=True, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Mean Squared Error and R2
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R2:", r2_score(y_test, y_pred, sample_weight=None))
df = pd.read_excel('predicted_data_with_inputs_BG_RF.xlsx')
df.head()
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
