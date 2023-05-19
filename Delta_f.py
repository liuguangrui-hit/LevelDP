import numpy as np
import torch

from ExtractFeature import ExtractFeature
from util.DataSet import get_data


def cal_delta_f(dataset, data_path, attentions):
    data, label, shape = get_data(dataset, data_path)
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    if isinstance(label, torch.Tensor):
        label = label.numpy()
    data = data.reshape(data.shape[0], -1)
    unique_label = np.unique(label)
    feature_level = np.load(f"attention/features/{dataset}.npy", allow_pickle=True)
    ans = np.zeros((7, attentions.shape[0]))
    for l in range(len(unique_label)):
        levels = feature_level[:, l]
        mask = label == l
        sub_set = data[mask]
        ratio = len(sub_set)/len(data)
        if dataset == "cicids" and len(sub_set) > 1000:
            np.random.shuffle(sub_set)
            sub_set = sub_set[:1000]
        for level_idx, level in enumerate(levels):
            ma = 0
            sub_sub_set = sub_set[:, level]
            for i in range(len(sub_sub_set) - 1):
                for j in range(i + 1, len(sub_sub_set)):
                    a = sub_sub_set[i]
                    b = sub_sub_set[j]
                    su = np.sum(np.abs(a - b) * attentions[l, level])
                    if su > ma:
                        ma = su
            ans[level_idx, l] = ma
        level = levels[0] + levels[1]
        ma = 0
        sub_sub_set = sub_set[:, level]
        for i in range(len(sub_sub_set) - 1):
            for j in range(i + 1, len(sub_sub_set)):
                a = sub_sub_set[i]
                b = sub_sub_set[j]
                su = np.sum(np.abs(a - b) * attentions[l, level])
                if su > ma:
                    ma = su
        ans[4, l] = ma
        level = levels[2] + levels[1]
        ma = 0
        sub_sub_set = sub_set[:, level]
        for i in range(len(sub_sub_set) - 1):
            for j in range(i + 1, len(sub_sub_set)):
                a = sub_sub_set[i]
                b = sub_sub_set[j]
                su = np.sum(np.abs(a - b) * attentions[l, level])
                if su > ma:
                    ma = su
        ans[5, l] = ma
        level = levels[0] + levels[1] + levels[2] + levels[3]
        ma = 0
        sub_sub_set = sub_set[:, level]
        for i in range(len(sub_sub_set) - 1):
            for j in range(i + 1, len(sub_sub_set)):
                a = sub_sub_set[i]
                b = sub_sub_set[j]
                su = np.sum(np.abs(a - b) * attentions[l, level])
                if su > ma:
                    ma = su
        ans[6, l] = ma
    np.save(f"delta_f/{dataset}.npy", ans)


if __name__ == "__main__":
    datasets = ["mnist", "ustc", "cicids"]
    for dataset in datasets:
        print(f"processing {dataset}")
        ef = ExtractFeature()
        config = ef.datasets.get(dataset)
        if dataset == "cicids":
            cal_delta_f(dataset, config["data_path"], ef.load_attentions(dataset, 7))
        else:
            cal_delta_f(dataset, config["data_path"], ef.load_attentions(dataset))
