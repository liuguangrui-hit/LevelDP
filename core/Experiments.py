import torchvision.transforms as T

from core.Models import MLP, DNN, CNN
from util.DataSet import CICIDS_2017, USTC_TFC2016, MNIST_


config_dict = {
    "mnist": {
        "data_set": MNIST_,
        "shadow_num": 5,
        "data_path": "mnist/MNIST/tmp/",
        "normal_label": None,
        "exp": {
            "MLP": {
                "model": MLP,
                "lr": 0.001,
                "epoch": 500,
                "bs": 500,
                "in_size": 784
            },
            "DNN": {
                "model": DNN,
                "lr": 0.001,
                "epoch": 500,
                "bs": 500,
                "in_size": 784
            },
            "CNN": {
                "model": CNN,
                "lr": 0.0001,
                "epoch": 500,
                "bs": 200,
                "in_size": 1,  # chanel num
                "transform": T.ToTensor()
            },
        }
    },

    "ustc": {
        "data_set": USTC_TFC2016,
        "shadow_num": 5,
        "data_path": "ustc/MNIST/tmp/",
        "normal_label": [0, 1, 2, 3, 5, 8, 9],
        "exp": {
            "MLP": {
                "model": MLP,
                "lr": 0.001,
                "epoch": 500,
                "bs": 500,
                "in_size": 784
            },
            "DNN": {
                "model": DNN,
                "lr": 0.001,
                "epoch": 500,
                "bs": 500,
                "in_size": 784
            },
            "CNN": {
                "model": CNN,
                "lr": 0.0001,
                "epoch": 500,
                "bs": 200,
                "in_size": 1,  # chanel num
                "transform": T.ToTensor()
            },
        }
    },

    "cicids": {
        "data_set": CICIDS_2017,
        "shadow_num": 5,
        "data_path": "cicids/",
        "normal_label": [0],
        "class_num": 7,
        "npy": True,
        "exp": {
            "MLP": {
                "model": MLP,
                "lr": 0.01,
                "epoch": 50,
                "bs": 5000,
                "in_size": 78
            },
            "DNN": {
                "model": DNN,
                "lr": 0.001,
                "epoch": 50,
                "bs": 5000,
                "in_size": 78
            },
        }
    }
}

