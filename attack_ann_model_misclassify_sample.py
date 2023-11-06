from tensorflow import keras

import numpy as np
import pandas as pd

# Load the model
model_stin = keras.models.load_model('saved-rf-sfs-ann.keras')

df = pd.read_csv('cleaned_data.csv')

labels = { 0: "Botnet", 1: "DDoS", 2: "Syn_DDoS", 3: "UDP_DDoS" }

df = df[df.iloc[:, -1] == 0]
sample = np.array(df.iloc[0])

sample_label = sample[-1]
sample = np.delete(sample, 10)

# Model expects 2-dim data. Transform into 2-D array that contains 1 sample.
sample = np.reshape(sample, (1,-1))

print(f"Original sample: {sample}")

sample_prediction = model_stin.predict(sample)

sample[0][8] = 1

print(f"Modified sample: {sample}")

modified_sample_prediction = model_stin.predict(sample)

def get_class_from_prediction(prediction):
    return labels[np.argmax(prediction)]

print(f"Real sample label: {get_class_from_prediction(labels[int(sample_label)])}")
print(f"Unmodified sample prediction: {get_class_from_prediction(sample_prediction)}")
print(f"Modified sample prediction: {get_class_from_prediction(modified_sample_prediction)}")
