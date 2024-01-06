"""
==================
3. ANN for BG dye
==================
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor, MLPClassifier
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

data = pd.read_excel("BG.xlsx")
data.head()
X = data.drop(columns=['Stirringspeed', 'Temp', 'Time', 'Dosage', 'pH', 'Concentration'], axis=1)
y = data[['Concentration,Cf(mg/L)', 'Adsorption capacity(mg/g)', 'Adsorption efficiency(%)']]
X.columns = X.columns.astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

model = MLPRegressor(hidden_layer_sizes=(20, 20), activation='relu', solver='adam', max_iter=500, tol=1e-4, 
                     validation_fraction=0.15)
model.fit(X_train, y_train)


y_train = model.predict(X_train)
y_pred = model.predict(X_test)

# Mean Squared Error and R2
mse = mean_squared_error(y_test, y_pred)
print(y_pred)
print("Mean Squared Error:", mse)
print("R2:", r2_score(y_test, y_pred, sample_weight=None, force_finite=True))


df = pd.read_excel('predicted_data_with_inputs_BG_ANN.xlsx')
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
