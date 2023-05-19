import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Laplace

input_size = 1
hidden_size = 128
n_step = 784
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self, in_size=1, classes_num=10, return_h=False):
        if classes_num is None:
            classes_num = 10
        self.classes_num = classes_num
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(  # 输入1*28*28
            nn.Conv2d(in_size, 32, kernel_size=3, padding=1),  # 32*28*28
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32*14*14
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64*14*14
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*7*7
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128*7*7
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256*7*7
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256*7*7
            nn.MaxPool2d(kernel_size=3, stride=2),  # 256*3*3
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, classes_num)


        # 加噪参数
        self.labels = None
        self.cifar = False
        self.noise_features = None
        self.noise_scale = None
        self.m = None
        self.return_noise = False

    def forward(self, x):
        if self.labels is not None:
            x_ori = x.clone()
            for i in range(10):
                noise_features = self.noise_features[i]
                if self.cifar:
                    noise_features = np.concatenate(
                        [noise_features, noise_features + 1024, noise_features + 2048])
                mask = self.labels == i
                d = x[mask]
                noise = self.m.sample(d.shape).to(device)
                shape = d.shape
                if self.cifar:
                    noise = noise.reshape((-1, 3 * 32 * 32))
                    d = d.reshape((-1, 3 * 32 * 32))
                else:
                    noise = noise.reshape((-1, 28 * 28))
                    d = d.reshape((-1, 28 * 28))
                d[:, noise_features] += noise[:, noise_features] * self.noise_scale
                d = d.reshape(shape)
                x[mask] = d
            noise = x - x_ori
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        if self.return_noise:
            return x, noise
        return x



class DNN(torch.nn.Module):
    def __init__(self, in_size=n_step, classes_num=10, return_h=False):
        if classes_num is None:
            classes_num = 10
        self.classes_num = classes_num
        super(DNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            # nn.Linear(1024, 1024),
            # nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, classes_num)
        )

        # 加噪的参数
        self.labels = None
        self.cifar = False
        self.noise_features = None
        self.noise_scale = None
        self.m = None
        self.return_noise = False

    def forward(self, x):
        if self.labels is not None:
            x_ori = x.clone()
            noise_features = self.noise_features
            for i in range(10):
                mask = self.labels == i
                d = x[mask]
                noise = self.m.sample(d.shape).to(device)
                # noise = torch.abs(noise)
                d[:, noise_features[i]] += noise[:, noise_features[i]] * self.noise_scale
                x[mask] = d
            noise = x - x_ori
        x = self.model(x)
        if self.return_noise:
            return x, noise
        return x


class MLP(nn.Module):
    def __init__(self, in_size=n_step, classes_num=10, noise=False, delta_f=None, features=None, return_h=False):
        if classes_num is None:
            classes_num = 10
        self.classes_num = classes_num
        super(MLP, self).__init__()
        self.best = 0
        self.model = nn.Sequential(
            nn.Linear(in_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, classes_num)
        )
        self.noise = noise
        if delta_f is not None:
            self.laplace = torch.distributions.Laplace(0, torch.tensor(delta_f))
        self.features = features

        # 加噪参数
        self.labels = None
        self.cifar = False
        self.noise_features = None
        self.noise_scale = None
        self.m = None

        self.return_noise = False

    def forward(self, x):
        if self.labels is not None:
            x_ori = x.clone()
            noise_features = self.noise_features
            for i in range(10):
                mask = self.labels == i
                d = x[mask]
                noise = self.m.sample(d.shape).to(device)
                # noise = torch.abs(noise)
                d[:, noise_features[i]] += noise[:, noise_features[i]] * self.noise_scale
                x[mask] = d
            noise = x - x_ori
        x = self.model(x)
        if self.return_noise:
            return x, noise
        return x

    def set_delta_f(self, delta_f):
        self.laplace = torch.distributions.Laplace(0, torch.tensor(delta_f))


class LSTM(torch.nn.Module):
    def __init__(self, in_size=input_size, classes_num=10, return_h=False):
        if classes_num is None:
            classes_num = 10
        self.classes_num = classes_num
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=in_size, hidden_size=hidden_size)
        self.linear = nn.Linear(hidden_size, classes_num)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        bs = x.shape[0]
        x = x.transpose(0, 1)
        hidden_state = torch.zeros(1, bs, hidden_size).to(self.device)
        cell_state = torch.zeros(1, bs, hidden_size).to(self.device)
        output, _ = self.lstm(x, (hidden_state, cell_state))
        output = output[-1]
        output = self.linear(output)
        return output


class Attacker(nn.Module):
    def __init__(self, classes_num=10, return_h=False):
        super(Attacker, self).__init__()
        if classes_num is None:
            classes_num = 10
        self.model = nn.Sequential(
            nn.Linear(classes_num, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.sigmoid = nn.Sigmoid()
        self.return_h = return_h

    def forward(self, x):
        x = self.model(x)
        x_ = self.sigmoid(x)
        if self.return_h:
            return x_, x
        return x_


def load_model(model, path: str):
    dt = torch.load(path)
    st = dt["state_dict"]
    model.load_state_dict(st)
    return model


def train(model, data_loader, lr=0.001, epochs=500, save_path=None, visible=False, device=torch.device("cuda"),
          dp=None, l2=0):
    '''
    :param model: 要训练的模型
    :param data_loader: (train_loader,test_loader)
    :param lr: 学习率
    :param epochs: 训练轮数
    :param save_path: 模型参数保存地址
    :param visible: 如果为true则打印训练信息，并保存tensorboard的信息
    :param device: torch.device 训练模型的设备
    :return: 训练好的模型
    '''
    import time
    if isinstance(data_loader, tuple):
        train_loader, test_loader = data_loader
    else:
        train_loader, test_loader = data_loader, None
    if test_loader is None:
        visible = False
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    s_time = time.time()
    for epoch in range(epochs):
        total_loss = 0
        for data, label in train_loader:
            if dp is not None and dp["t"] == "i":
                if dp.get("mask") is not None:
                    mask = dp.get("mask")
                    if len(dp["mask"][0]) != 0:
                        for c_idx in range(model.classes_num):
                            m = Laplace(torch.tensor([0.0]), torch.tensor(dp["s"][c_idx]))
                            d = data[label == c_idx]
                            shape = d.shape
                            d = d.reshape((shape[0], -1))
                            features = mask[c_idx]
                            noise = m.sample((len(d), len(features)))
                            d[:, features] += noise.squeeze(-1)
                            d = d.reshape(shape)
                            data[label == c_idx] = d
                else:
                    for c_idx in range(model.classes_num):
                        m = Laplace(torch.tensor([0.0]), torch.tensor(dp["s"][c_idx]))
                        d = data[label == c_idx]
                        noise = m.sample(d.shape)
                        d += noise
                        data[label == c_idx] = d
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, label)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        if test_loader is not None:
            with torch.no_grad():
                correct = 0
                total = 0
                test_model = model
                for data, label in test_loader:
                    data, label = data.to(device), label.to(device)
                    out = test_model(data)
                    total += len(label)
                    correct += torch.sum(torch.argmax(out, dim=1) == label).item()
                accuracy = correct / total
                if visible:
                    print(f"epoch: {epoch}")
                    print(f"\t准确率为：{accuracy}")
                    print(f"\t损失为：{total_loss}")
        if visible:
            e_time = time.time()
            print(f"time:{e_time - s_time}")
            s_time = e_time
    if save_path is not None:
        torch.save({"state_dict": model.state_dict()}, save_path)
    return model
