import trustee
import pandas as pd
import numpy as np

import pickle

from trustee.utils import log
from trustee.report.trust import TrustReport

# Load the model
with open('saved-rf-sfs-ann.pkl', 'rb') as f:
    model = pickle.load(f)

OUTPUT_PATH = "/home/f646b388/rf-sfs-ann/res/output"
REPORT_PATH = f"{OUTPUT_PATH}/report/trust_report.obj"

dict_2class = { 0: "Botnet", 1: "DDoS", 2: "Syn_DDoS", 3: "UDP_DDoS" }
class_names = dict_2class.values()

logger = log.Logger(f"{OUTPUT_PATH}/output.log")

df = pd.read_csv('cleaned_data.csv', index_col=False)

# df = df.sample(n=100)

y = df['Label']
X = df
X.drop(columns='Label', inplace=True)

print(X.head())

class ModelWrapper():
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        print(X.shape)
        arr = X.to_numpy()
        return np.array([np.argmax(i) for i in self.model.predict(arr)])

X = X.to_numpy()
y = y.to_numpy()
y = np.reshape(y, (len(y),))
print(y.shape)
print(y)

wrapped_model = ModelWrapper(model)

trust_report = TrustReport(
    wrapped_model,
    X,
    y,
    top_k=10,
    max_iter=0,
    trustee_num_iter=10,
    num_pruning_iter=30,
    trustee_sample_size=0.3,
    analyze_stability=True,
    analyze_branches=True,
    skip_retrain=True,
    class_names=list(class_names),
    logger=logger,
    verbose=False,
)

trust_report.save(OUTPUT_PATH)
logger.log(trust_report)

# analyze trust report
# modify sample to force model to misclassify
