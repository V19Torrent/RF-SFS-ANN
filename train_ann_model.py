import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import pickle

# Data preparation
df_sat = pd.read_csv("SAT20.csv")
df_ter = pd.read_csv("TER20.csv")

frames = [df_sat, df_ter]
df_concat = pd.concat(frames)
df_concat.drop(columns='id', inplace=True)

# print(df_concat.head())
print(df_concat.shape)

# sample the dataset for easier debugging
# df_concat = df_concat.sample(n=100)

# These columns give problems with NaNs after normalizing
df_concat = df_concat.drop(columns=['syn_cnt', 'urg_cnt', 'bw_psh_flag', 'fw_urg_flag', 'bw_urg_flag', 'fin_cnt', 'psh_cnt', 'ece_cnt', 'fw_byt_blk_avg', 'fw_pkt_blk_avg', 'fw_blk_rate_avg', 'bw_byt_blk_avg', 'bw_pkt_blk_avg', 'bw_blk_rate_avg'])
# print(df_concat.head())

# Data exploration
# labels = df_concat["Label"]
# df_concat.drop(columns='Label', inplace=True)
# correlation_matrix = df_concat.corr()
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
# plt.title('Correlation Plot')
# plt.savefig("corr_matrix.png")
# plt.clf()

# def plot_histogram(feature):
#     plt.hist(df_concat[feature].dropna(), bins=20, edgecolor='black')
#     plt.title(f'Histogram of {feature}')
#     plt.xlabel(feature)
#     plt.ylabel('Frequency')
#     plt.savefig(f"histograms/{feature}.png")
#     plt.clf()

# print(df_concat.head())
# for feature in df_concat.columns:
#     plot_histogram(feature)

# i = 0
# last_index = 0
# first_index = 0
# last_index_max = len(df_concat.columns)
# while last_index < len(df_concat.columns):
#     last_index = first_index + 4
#     if last_index > last_index_max:
#         last_index = last_index_max
#     data_to_plot = df_concat.iloc[:, first_index:last_index]
#     data_to_plot['Label'] = labels
#     print(data_to_plot.head())
#     pairplot = sns.pairplot(data_to_plot, hue="Label")
#     plt.savefig(f"pairplots/pairplot{i}.png")
#     plt.clf()
#     first_index = last_index
#     i += 1
# Data exploration

# Minority removal (merge data)
labels = {
    'Syn_DDoS': 'Syn_DDoS',
    'UDP_DDoS': 'UDP_DDoS',
    'Botnet': 'Botnet',
    'Portmap_DDoS': 'DDoS',
    'Backdoor': 'Botnet',
    'Web Attack': 'Botnet',
    'LDAP_DDoS': 'DDoS',
    'MSSQL_DDoS': 'DDoS',
    'NetBIOS_DDoS': 'DDoS'
}
df_concat['Label'] = df_concat['Label'].map(labels)
print(df_concat['Label'].value_counts())

# Extract the label column
y = df_concat['Label']
df_concat.drop(columns='Label', inplace=True)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Print the mapping of original labels to encoded integers
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label mapping:", label_mapping)

# min max normalize
normalized_df=(df_concat-df_concat.min())/(df_concat.max()-df_concat.min())
print(normalized_df.head())

X = normalized_df

sfs = SFS(RandomForestClassifier(max_depth=2, random_state=0),
          k_features=10,
          forward=True,
          floating=False,
          scoring = 'accuracy',
          cv = 0)
sfs.fit(X,y_encoded)
print(sfs.k_feature_names_)

X = X[np.array(sfs.k_feature_names_)]
# Data prep complete, X and y_encoded are the variables to train the models with

def build_ann_model(input_dim, hidden_neurons, output_neurons, activation_hidden='tanh', activation_output='softmax'):
    model = Sequential()

    # Input layer
    model.add(Dense(hidden_neurons, input_dim=input_dim, activation=activation_hidden))

    # Hidden layers (3 layers with hidden_neurons neurons each)
    for _ in range(3):
        model.add(Dense(hidden_neurons, activation=activation_hidden))

    # Output layer
    model.add(Dense(output_neurons, activation=activation_output))

    return model



# Parameters from your description
input_dim = 10  # Input dimension for STIN datasets
hidden_neurons = 50  # Number of neurons in each hidden layer
output_neurons_stin = 4  # Number of output neurons for STIN dataset

# Build the ANN models for STIN datasets
model_stin = build_ann_model(input_dim, hidden_neurons, output_neurons_stin)

# Compile the models
adam_optimizer = Adam()  # Adam optimizer
model_stin.compile(optimizer=adam_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the models
n_epochs = 5  # Set the number of epochs

X_train_stin, X_test_stin, y_train_stin, y_test_stin = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("Shape of X_train_stin:", X_train_stin.shape)
print("Shape of X_test_stin:", X_test_stin.shape)
print("Shape of y_train_stin:", y_train_stin.shape)
print("Shape of y_test_stin:", y_test_stin.shape)

# Train for STIN dataset
model_stin.fit(X_train_stin, y_train_stin, epochs=n_epochs)

y_predictions = model_stin.predict(X_test_stin)

# Save the model
with open('saved-rf-sfs-ann.pkl', 'wb') as f:
    pickle.dump(model_stin, f)
