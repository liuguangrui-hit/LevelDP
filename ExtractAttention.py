import argparse

import numpy as np
from core.SAN import SAN
import seaborn as sns
import torch
from scipy import sparse

sns.set_style("whitegrid")
parser = argparse.ArgumentParser(description='Feature importance extraction.')
parser.add_argument('--dataset', type=str, default="mnist")
parser.add_argument('--epoch', type=int, default="500")

args = parser.parse_args()

epochs = args.epoch
dataset = args.dataset
# mnist
if dataset == "mnist":
    data_path = r"mnist/MNIST/tmp/train.pt"
    feature_num = 28 * 28
    X, Y = torch.load(data_path)
    X = X / 255
    l = len(X)
    X = X.reshape((l, feature_num))
elif dataset == "ustc":
    data_path = r"ustc/MNIST/tmp/train.pt"
    feature_num = 28 * 28
    X, Y = torch.load(data_path)
    X = X / 255
    l = len(X)
    X = X.reshape((l, feature_num))
elif dataset == "cicids":
    X = np.load("cicids/train.npy")
    Y = np.load("cicids/train.npy")[:, -1].astype(int)
    X = X[:, :-1]
    feature_num = 78
    min = np.min(X, axis=0)
    max = np.max(X, axis=0)
    X = (X - min) / (max + 0.0001)
else:
    print("invalid dataset")
    import sys
    sys.exit()

names = [i for i in range(feature_num)]

clf = SAN(num_epochs=epochs, num_heads=5, batch_size=500, dropout=0, hidden_layer_size=512, stopping_crit=20,
              learning_rate=0.01)
clf.fit(sparse.csr_matrix(X), Y)
preds = clf.predict(sparse.csr_matrix(X))
for label in np.unique(Y):
    data = X[Y == label]
    l = Y[Y == label]
    data = sparse.csr_matrix(data)
    pred = clf.predict(data)
    local_attention_matrix = clf.get_instance_attention(data)
    mean = np.mean(local_attention_matrix, axis=0)
    np.save(f"attention/attention/cicids/{label}.npy", mean)
print("done")
