import os

import torch
import torchvision
from torchvision.datasets import MNIST

name = os.name
USTC_root = "../ustc/"

if __name__ == "__main__":
    train = MNIST(USTC_root, train=True, transform=torchvision.transforms.ToTensor(),
                  download=False)

    indexs = torch.randperm(len(train))
    data = train.data[indexs]
    label = train.targets[indexs]

    num = 1000
    train_x = data[:num]
    train_y = label[:num]
    torch.save((train_x, train_y), os.path.join(os.path.join(os.path.join(USTC_root, "MNIST"), "tmp"), "train.pt"))
    test_x = data[num:2 * num]
    test_y = label[num:2 * num]
    torch.save((test_x, test_y), os.path.join(os.path.join(os.path.join(USTC_root, "MNIST"), "tmp"), "test.pt"))

    train_x = data[2 * num:3 * num]
    train_y = label[2 * num:3 * num]
    torch.save((train_x, train_y),
               os.path.join(os.path.join(os.path.join(USTC_root, "MNIST"), "tmp"), "shadow_train_0.pt"))
    test_x = data[3 * num:4 * num]
    test_y = label[3 * num:4 * num]
    torch.save((test_x, test_y),
               os.path.join(os.path.join(os.path.join(USTC_root, "MNIST"), "tmp"), "shadow_test_0.pt"))


