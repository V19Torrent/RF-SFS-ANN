from tensorflow import keras

import numpy as np
import pandas as pd

import sklearn.metrics as metrics
import seaborn as sns

import matplotlib.pyplot as plt

# Load the model
model_stin = keras.models.load_model('saved-rf-sfs-ann.keras')

df = pd.read_csv('cleaned_data.csv')

# Uncomment when debugging
# df = df.sample(n=100)

y = df['Label']
df.drop(columns="Label", inplace=True)
X = df

X.head()

predictions = model_stin.predict(X)

cm_good = metrics.confusion_matrix(y, np.argmax(predictions, axis=1))

accuracy_clean_data = metrics.accuracy_score(y, np.argmax(predictions, axis=1)) * 100

X['fw_win_byt'] = 1

predictions = model_stin.predict(X)
accuracy_affected_data = metrics.accuracy_score(y, np.argmax(predictions, axis=1)) * 100

cm_bad = metrics.confusion_matrix(y, np.argmax(predictions, axis=1))

print(f"Accuracy using unmodified data {accuracy_clean_data:.2f}%")
print(f"Accuracy using modified data {accuracy_affected_data:.2f}%")

# Data for the bars
categories = ['Unmodified Data', 'Modified Data']
values = [accuracy_clean_data, accuracy_affected_data]  # These values should represent percentages

# Create a figure and axis
fig, ax = plt.subplots()

# Set the y-axis limit to 100% and the x-axis limit to 0%
ax.set_ylim(0, 100)

# Create the bar chart
bars = ax.bar(categories, values, color=['green', 'red'])

# Add labels to the bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}%', (bar.get_x() + bar.get_width() / 2, height), ha='center', va='bottom')

# Add labels and title
ax.set_xlabel('Categories')
ax.set_ylabel('Percentage')
ax.set_title('Attack Accuracy')

# Show the chart
plt.savefig('acc.png')
plt.clf()

sns.heatmap(cm_good,
            annot=True,
            fmt='g')
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix Good Data',fontsize=17)
plt.savefig("confusion_matrix_good.png")
plt.clf()

sns.heatmap(cm_bad,
            annot=True,
            fmt='g')
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix Bad Data',fontsize=17)
plt.savefig("confusion_matrix_bad.png")
