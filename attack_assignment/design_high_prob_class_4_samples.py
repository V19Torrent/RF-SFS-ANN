from tensorflow import keras
import pandas as pd
import numpy as np

import h5py
import time

# Load the model
model = keras.models.load_model('model24.keras')

base_sample = [0.06158, 0.02082, 0.30915, 0.28980, 0.03152, 0.30347, 0.18008, 0.46782, 0.60911, 0.81574]

prediction = model.predict([base_sample])

print(prediction)

modified_feature_to_samples = {}

step = 0.001
value_to_set_feature = [i * step for i in range(int(1/step) + 1)]
for i in range(0, len(base_sample)):
    modified_feature_to_samples[i] = []
    for j in value_to_set_feature:
        modified_sample = base_sample[:]
        modified_sample[i] = j

        prediction = model.predict([modified_sample], verbose=0)

        if prediction[0][3] > 0.9:
            print(prediction)
            print(f"Modified feature {i} to {j}")
            modified_feature_to_samples[i].append(modified_sample[:])

with open("output.txt", 'w') as f:
    for modified_samples in modified_feature_to_samples:
        f.write(f"New samples for modifying feature {modified_samples}\n")
        for sample in modified_feature_to_samples[modified_samples]:
            f.write(", ".join([str(round(sample, 6)) for sample in sample]) + "\n")

    total_new_samples = 0
    for modified_samples in modified_feature_to_samples:
        new_samples = len(modified_feature_to_samples[modified_samples])
        total_new_samples += new_samples
        f.write(f"Samples after modifying feature {modified_samples}: {new_samples}\n")

    f.write(f"Total new samples with greater than 90% confidence: {total_new_samples}\n")

