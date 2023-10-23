import pickle

# Load the model
with open('saved-rf-sfs-ann.pkl', 'rb') as f:
    model_stin = pickle.load(f)

# model_stin.predict(XYZ)
