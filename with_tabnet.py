"""
====================
8. TabNet
====================
"""

import os
import random
import tensorflow as tf
import numpy as np

np.random.seed(1575)
random.seed(1575)
os.environ['PYTHONHASHSEED'] = str(1575)
tf.random.set_seed(1575)

from tabnet.tabnet import TabNetRegressor

from ai4water.postprocessing import LossCurve
from SeqMetrics import RegressionMetrics

from utils import get_dataset

# %%

ads_df_enc ,  _, _ = get_dataset()

# %%

X_train, y_train = ads_df_enc.training_data()
X_valid, y_valid = ads_df_enc.validation_data()
X_test, y_test = ads_df_enc.test_data()

# %%

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1000, seed=1575,
                                      reshuffle_each_iteration=True)
train_dataset = train_dataset.batch(64).prefetch(tf.data.experimental.AUTOTUNE)

# %%

valid_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
valid_dataset = valid_dataset.shuffle(buffer_size=1000, seed=1575,
                                      reshuffle_each_iteration=True)
valid_dataset = valid_dataset.batch(64).prefetch(tf.data.experimental.AUTOTUNE)

# %%

feature_columns = []
for col_name in ads_df_enc.input_features:
    feature_columns.append(tf.feature_column.numeric_column(col_name))

# %%

clf = TabNetRegressor( feature_columns=None, num_features=74,
                        num_regressors=1, output_dim=1,
                        # optimizer_params={'lr': 0.009578661171988185}
                      )

# %%

lr = tf.keras.optimizers.schedules.ExponentialDecay(0.01,
                                                    decay_steps=100,
                                                    decay_rate=0.9,
                                                    staircase=False)

# %%

optimizer = tf.keras.optimizers.Adam(0.0001)

# %%

clf.compile(optimizer, loss='mse', #metrics=['nse']
            )

# %%

h = clf.fit(train_dataset, epochs=5,
            validation_data=valid_dataset,
        verbose=2)

# %%

LossCurve().plot_loss(h.history)

# %%

preds = clf.predict(X_test)

# %%

print(f'test r2 score = {RegressionMetrics(y_test,preds).r2_score()}')

print(f'test r2 = {RegressionMetrics(y_test,preds).r2()}')

print(f'test mse = {RegressionMetrics(y_test,preds).mse()}')
