from tensorflow import keras
import pandas as pd
import numpy as np

import h5py
import time

# Load the model
model = keras.models.load_model('model24.keras')

print(model.summary())

def generate_samples():
    # Set seed every time
    seed = int(time.time())
    print(f"Here is the seed: {seed}")
    np.random.seed(seed)

    # Define the number of rows and columns
    num_rows = 1000
    num_columns = 10

    # Create a dataframe with random values
    data = np.random.rand(num_rows, num_columns)

    # Create column names
    columns = [f'Column_{i}' for i in range(1, num_columns + 1)]

    # Create the dataframe
    return pd.DataFrame(data, columns=columns)

samples = pd.DataFrame()
probabilityThreshold = 0.9
found = False
selected_rows = []
mask = []
while not found:
    # Generate dataset
    samples = generate_samples()

    # Save to file
    samples.to_csv("randomized_dataset.csv", index=False)

    # Make predictions
    predictions = model.predict(samples)

    # Filter the predictions
    mask = predictions[:, 3] >= probabilityThreshold
    selected_rows = predictions[mask]
    print(selected_rows)

    # If we have at least one row that was filtered, we are done
    if len(selected_rows) >= 1:
        found = True

selected_sample_indices = np.where(mask)
selected_samples = samples.iloc[selected_sample_indices]
np.savetxt("indices_of_selected_rows.txt", selected_sample_indices, fmt='%.2f', delimiter=', ')
np.savetxt("selected_samples.txt", selected_samples, fmt='%.5f', delimiter=', ')
