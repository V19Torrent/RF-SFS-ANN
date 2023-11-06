import pickle

import numpy as np
import pandas as pd

import sklearn.metrics as metrics

# Load the model
with open('saved-rf-sfs-ann.pkl', 'rb') as f:
    model_stin = pickle.load(f)

df = pd.read_csv('cleaned_data.csv')

# Uncomment when debugging
df = df.sample(n=100)

y = df['Label']
df.drop(columns="Label", inplace=True)
X = df

X.head()

predictions = model_stin.predict(X)

accuracy_clean_data = metrics.accuracy_score(y, np.argmax(predictions, axis=1)) * 100

X['fw_win_byt'] = 1

predictions = model_stin.predict(X)
accuracy_affected_data = metrics.accuracy_score(y, np.argmax(predictions, axis=1)) * 100

print(f"Accuracy using unmodified data {accuracy_clean_data:.2f}%")
print(f"Accuracy using modified data {accuracy_affected_data:.2f}%")
