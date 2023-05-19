import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class USTC_TFC2016(torchvision.datasets.MNIST):
    def __init__(self, kind="train", transform=T.Compose(
        [T.ToTensor(), T.Lambda(lambda x: torch.flatten(x))]),
                 target_transform=None, min=None, max=None, data_path=None, data=None, label=None):
        self.root = os.path.expanduser("ustc/MNIST/")
        self.transform = transform
        self.target_transform = target_transform
        self.train = False  # training set or test set
        self.min = min
        self.max = max
        self.no_PIL = False
        if data is not None:
            self.data, self.targets = data, label
            self.no_PIL = True
        else:
            if data_path is None:
                self.data, self.targets = torch.load(os.path.join(self.tmp_folder, f"{kind}.pt"))
            else:
                self.data, self.targets = torch.load(data_path)

    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])

        if not self.no_PIL:
            img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    @property
    def tmp_folder(self):
        return os.path.join(self.root, "tmp")


class MNIST_(torchvision.datasets.MNIST):
    def __init__(self, kind="train", transform=T.Compose(
        [T.ToTensor(), T.Lambda(lambda x: torch.flatten(x))]),
                 target_transform=None, min=None, max=None, data_path=None, data=None, label=None):
        self.root = os.path.expanduser("mnist/MNIST/")
        self.transform = transform
        self.target_transform = target_transform
        self.train = False  # training set or test set
        self.min = min
        self.max = max
        self.no_PIL = False
        if data is not None:
            self.data, self.targets = data, label
            self.no_PIL = True
        else:
            if data_path is None:
                self.data, self.targets = torch.load(os.path.join(self.tmp_folder, f"{kind}.pt"))
            else:
                self.data, self.targets = torch.load(data_path)

    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])

        if not self.no_PIL:
            img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    @property
    def tmp_folder(self):
        return os.path.join(self.root, "tmp")


class CICIDS_2017(Dataset):
    def __init__(self, kind="train", min=None, max=None, unsqueeze=False, cnn=False, data=None, label=None):
        super(CICIDS_2017, self).__init__()
        self.data = None
        self.label = None
        self.kind = kind
        self.min = min
        self.max = max
        self.unsqueeze = unsqueeze
        self.cnn = cnn
        if data is None:
            self._load_data()
        else:
            self.data, self.label = data, label

    def _load_data(self):
        self.data, self.label, self.mean, self.max = get_cicids_data(self.kind, self.min, self.max)

    def __getitem__(self, item):
        if self.unsqueeze:
            d = torch.Tensor(self.data[item]).unsqueeze(-1)
        elif self.cnn:
            d = torch.Tensor(self.data[item].reshape((1, 6, 13)))
        else:
            d = torch.Tensor(self.data[item])
        l = torch.Tensor([self.label[item]]).squeeze(0).long()
        return d, l

    def __len__(self):
        if self.data is None:
            return 0
        return len(self.data)


def process_cicids(name, save_name):
    path = os.path.join("cicids/", name)
    data = np.loadtxt(path, float, delimiter=",", skiprows=1, usecols=list(range(78)),
                      encoding="UTF-8")
    label = np.loadtxt(path, str, delimiter=",", skiprows=1, usecols=[78],
                       encoding="UTF-8")
    f_d, f_l = None, None
    for idx, k in enumerate(class_dict.keys()):
        mask = label == k
        d = data[mask]
        l = label[mask]
        l = l[:len(d) // 4]
        d = d[:len(d) // 4]
        l[:] = str(class_dict.get(k))
        if f_d is None:
            f_d = d
            f_l = l
        else:
            f_d = np.row_stack((f_d, d))
            f_l = np.concatenate((f_l, l))
    data = f_d
    label = f_l
    label = label.astype(float)
    data = np.column_stack((data, label))
    np.save(os.path.join("cicids/", f"{save_name}.npy"), data)


def get_cicids_data(name, min_, max_):
    path = os.path.join("cicids/", f"{name}.npy")
    data = np.load(path)
    label = data[:, -1].astype(int)
    data = data[:, :-1]
    if min_ is None:
        min_ = np.min(data, axis=0)
        max_ = np.max(data, axis=0)
    data = (data - min_) / (max_ - min_ + 1e-6)
    return data, label, min_, max_


class_dict = {'BENIGN': 0, 'Bot': 1, 'Bruteforce': 2, 'DDoS': 3, 'Dos': 4, 'PortScan': 5, 'Web-Attack': 6}


def get_data(dataset, data_path, name="train", transform=None, cnn=False, unsqueeze=False):
    if dataset == "cicids":
        data = np.load(os.path.join(data_path, f"{name}.npy"))
        label = torch.tensor(data[:, -1])
        data = torch.tensor(data[:, :-1])
        ma = torch.max(data, dim=0).values
        mi = torch.min(data, dim=0).values
        data = (data - mi) / (ma - mi + 1e-6)
    elif dataset == "purchase":
        data, label = torch.load(os.path.join(data_path, f"{name}.pt"))
        data, label = torch.tensor(data), torch.tensor(label)
    else:
        data, label = torch.load(os.path.join(data_path, f"{name}.pt"))
        data, label = torch.tensor(data) / 255, torch.tensor(label)
    shape = data.shape
    if unsqueeze:
        data = data.unsqueeze(-1)
    # 挨个transform
    data = data.numpy()
    label = label.numpy()
    return data, label, shape


def get_data_set(exp_config, DS, key, exp, t_data=None, test_data=None, t_label=None, shadow=False):
    if shadow:
        shadow_prefix = "shadow_"
        shadow_suffix = "_fixed_0"
    else:
        shadow_prefix = ""
        shadow_suffix = ""
    if exp_config.get("transform") is not None:
        train_ = DS(shadow_prefix + "train" + shadow_suffix, data=t_data, label=t_label,
                    transform=exp_config["transform"])
        # train = DS(transform=exp_config["transform"])
        test_ = DS(shadow_prefix + "test" + shadow_suffix, min=train_.min, max=train_.max,
                   transform=exp_config["transform"], data=test_data,
                   label=t_label)
    else:
        if (key == "cicids" and exp == "LSTM") or (key == "purchase" and exp == "LSTM"):
            train_ = DS(shadow_prefix + "train" + shadow_suffix, data=t_data, label=t_label, unsqueeze=True)
            # train = DS(unsqueeze=True)
            test_ = DS(shadow_prefix + "test" + shadow_suffix, min=train_.min, max=train_.max, unsqueeze=True,
                       data=test_data, label=t_label)
        elif (key == "purchase" or key == "cicids") and exp == "CNN":
            train_ = DS(shadow_prefix + "train" + shadow_suffix, data=t_data, label=t_label, cnn=True)
            # train = DS(cnn=True)
            test_ = DS(shadow_prefix + "test" + shadow_suffix, min=train_.min, max=train_.max, cnn=True,
                       data=test_data, label=t_label)
        else:
            train_ = DS(shadow_prefix + "train" + shadow_suffix, data=t_data, label=t_label)
            # train = DS()
            test_ = DS(shadow_prefix + "test" + shadow_suffix, min=train_.min, max=train_.max, data=test_data,
                       label=t_label)
    return train_, test_


if __name__ == "__main__":
    os.chdir("../")
    process_cicids(name="1.0_test.csv", save_name="test")
    process_cicids(name="1.0_train.csv", save_name="train")
