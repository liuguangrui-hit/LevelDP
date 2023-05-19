import heapq
import os

import numpy as np
import torch


class ExtractFeature:
    def __init__(self):
        self.datasets = {
            "ustc": {"thresh": 1e-2, "data_path": "ustc/MNIST/tmp/"},
            "mnist": {"thresh": 1e-2, "data_path": "mnist/MNIST/tmp/"},
            "cicids": {"thresh": 1e-5, "data_path": "cicids/"},
        }
        self.attention_prefix = "attention/attention"

    def load_attentions(self, p, num=10):
        '''
        加载注意力权重
        :param p: 存储权重文件夹
        :param num: 类别数
        :return: 重要特征权重
        '''
        attention = None
        for i in range(num):
            a = np.load(os.path.join(self.attention_prefix, p, f"{i}.npy"))
            if attention is None:
                attention = a
            else:
                attention = np.row_stack((attention, a))
        return attention

    def parse_1(self, attentions, threshold, ratio=0.1) -> list:
        ans = []
        num = len(attentions)
        feature_num = attentions.shape[1]
        if ratio is not None:
            for i in range(num):
                a = heapq.nlargest(len(attentions[i]), range(len(attentions[i])), attentions[i].__getitem__)
                ans.append(a[:int(feature_num * ratio)])
            return ans

        rows, cols = np.where(attentions > threshold)
        for i in range(num):
            mask = rows == i
            col = cols[mask]
            col = col.tolist()
            col.sort(key=lambda x: attentions[i][x])
            ans.append(col)
        return ans

    def parse_2(self, from1, attentions) -> list:
        num = len(attentions)
        union = set()
        ans = []
        for i in range(num):
            union = union | set(from1[i])
        for i in range(num):
            a = list(union - set(from1[i]))
            a.sort(key=lambda x: attentions[i][x])
            ans.append(a)
        return ans

    def parse_3(self, from1, from2, from4, attentions) -> list:
        num = len(attentions)
        all = set(list(range(attentions.shape[1])))
        ans = []
        for i in range(num):
            a = ((all - set(from1[i])) - set(from2[i])) - set(from4[i])
            ans.append(list(a))
        return ans

    def parse_4(self, p, from1, from2, attention, threshold, ratio=0.3) -> list:
        ans = []
        # 加载训练数据
        if p == "cicids":
            data = np.load(os.path.join(self.datasets[p].get("data_path"), "train.npy"))
            label = data[:, -1]
            data = data[:, :-1]
            mi = data.min(axis=0)
            ma = data.max(axis=0)
            data = (data - mi) / (ma - mi + 1e-6)
        else:
            data, label = torch.load(os.path.join(self.datasets[p].get("data_path"), "train.pt"))
            data = (data / 255)
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        if isinstance(label, torch.Tensor):
            label = label.numpy()
        if p == "cifar":
            data = np.mean(data, axis=3)
        data = data.reshape(-1, attention.shape[1])

        # for 每一类，计算训练数据的各特征极差与该类该特征的注意力权重的乘积
        num_features = attentions.shape[1]
        for i in range(len(np.unique(label))):
            mask = label == i
            d = data[mask]
            d = d * attention[i]
            ma = d.max(axis=0)
            mi = d.min(axis=0)
            if ratio is not None:
                a = ma - mi
                a = heapq.nsmallest(len(a), range(len(a)), a.__getitem__)
                a = a[:int(num_features * ratio)]
                ans.append(list(set(a) - set(from1[i]) - set(from2[i])))
                continue
            ans.append(
                list(set(np.where((ma - mi) < threshold)[0].tolist()) - set(from1[i]) - set(from2[i])))
        return ans

    def check_conflict(self, from1, from2, from4):
        num = len(from1)
        for i in range(num):
            if len(set(from1[i]) | set(from2[i]) | set(from4)) < len(from1[i]) + len(from2[i]) + len(from4[i]):
                print(i)


if __name__ == "__main__":
    datasets = ["mnist", "ustc", "cicids"]
    for dataset in datasets:
        ef = ExtractFeature()
        config = ef.datasets.get(dataset)
        thresh = 0.0001
        if dataset == "cicids":
            attentions = ef.load_attentions(dataset, num=7)
        else:
            attentions = ef.load_attentions(dataset, num=10)
        f_1 = ef.parse_1(attentions, thresh)
        f_2 = ef.parse_2(f_1, attentions)
        f_4 = ef.parse_4(dataset, f_1, f_2, attentions, thresh)
        f_3 = ef.parse_3(f_1, f_2, f_4, attentions)
        np.save(f"attention/features/{dataset}.npy", np.asarray([f_1, f_2, f_3, f_4], dtype=object))
