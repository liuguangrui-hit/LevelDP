import argparse
import warnings

import numpy
import numpy as np
from torch.utils.data import DataLoader

from core.Attack import song_membership_inference, fix_label, \
    train_single_shadow_model, Ye_membership_inference_attack
from core.Experiments import config_dict
from core.Models import train as trainer
from util.DataSet import *
from util.DataSet import get_data, get_data_set

warnings.filterwarnings("ignore")

delta_f = None


def add_noise(ep, dataset, data_path):
    delta_f = np.load(f"delta_f/{dataset}.npy")[5] / ep  # (6,classes)
    data, label, shape = get_data(dataset, data_path)
    data = data.reshape(shape[0], -1)
    f = np.load(f"attention/features/{dataset}.npy", allow_pickle=True)
    for i in range(len(delta_f)):
        noised_features = f[1][i] + f[2][i]
        n_f_len = len(noised_features)
        noised_features = noised_features[:n_f_len]
        mask = label == i
        d = data[mask]  # 筛选某一类别特征
        num = len(d)
        m = torch.distributions.Laplace(0, torch.tensor([delta_f[i]]))
        noise = m.sample((num, len(noised_features))).squeeze(-1)  # 噪声采样
        if isinstance(d, numpy.ndarray):
            noise = noise.detach().cpu().numpy()
        d[:, noised_features] += noise
        data[mask] = d
    data = data.reshape(shape)
    return data, label


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Test our defense method.')
    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument('--model', type=str, default="DNN")
    parser.add_argument('--epsilon', type=float, default=1e-3)
    args = parser.parse_args()
    data_set = args.dataset
    model_name = args.model
    ep = args.epsilon

    print(f"dataset: {data_set}")
    config = config_dict.get(data_set)
    DS = config["data_set"]
    shadow_num = config["shadow_num"]
    exps = config["exp"]
    print(f"model: {model_name}")
    exp_config = exps[model_name]
    batch_size = exp_config["bs"]
    lr = exp_config["lr"]
    epochs = exp_config["epoch"]
    model_proto = exp_config["model"]
    train_data, train_label = add_noise(ep, data_set, data_path=config["data_path"])
    if isinstance(train_data, torch.Tensor):
        train_data = train_data.numpy()
    if isinstance(train_label, torch.Tensor):
        train_label = train_label.numpy()

    train, test = get_data_set(exp_config, DS, data_set, model_name, t_data=train_data, t_label=train_label)
    train_loader = DataLoader(train, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)
    if exp_config.get("in_size") is None:
        model = model_proto(classes_num=config.get("class_num")).to(device)
    else:
        model = model_proto(classes_num=config.get("class_num"), in_size=exp_config.get("in_size")).to(
            device)

    model = trainer(model, (train_loader, test_loader), lr, epochs,
                    save_path=None,
                    visible=False, device=device)

    train, _ = get_data_set(exp_config, DS, data_set, model_name)
    train_loader = DataLoader(train, batch_size=batch_size)
    pred_score = []
    l = []
    true_label = []
    with torch.no_grad():
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            out = model(data)
            out = torch.softmax(out, dim=1)
            pred_score.append(out.tolist())
            true_label.append(label.tolist())
            l.append([1] * len(out))
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            out = model(data)
            out = torch.softmax(out, dim=1)
            pred_score.append(out.tolist())
            true_label.append(label.tolist())
            l.append([0] * len(out))
    pred_score = np.concatenate(pred_score)
    l = np.concatenate(l)
    true_label = np.concatenate(true_label)
    acc = np.sum(np.argmax(pred_score, axis=1)[l == 0] == true_label[l == 0]) / len(true_label[l == 0])
    print(f"acc : {acc}")

    # shadow model training
    fix_label(model, f"shadow_train_{0}", save_path=f"{config['data_path']}shadow_train_fixed_{0}",
              dataset=DS,
              exp_config=exp_config, data_folder=config['data_path'], npy=config.get("npy"))
    fix_label(model, f"shadow_test_{0}", save_path=f"{config['data_path']}shadow_test_fixed_{0}",
              dataset=DS,
              exp_config=exp_config, data_folder=config['data_path'], npy=config.get("npy"))
    s_x, s_y, s_c, shadow_model = train_single_shadow_model(0, dataset=DS, exp_config=exp_config,
                                                            return_model=True,
                                                            classes_num=config.get("class_num"),
                                                            model_path=None)

    # song成员推断
    s_x, s_y, s_c = np.asarray(s_x), np.asarray(s_y), np.asarray(s_c)
    train_mask = s_y == 1
    shadow_train_performance = s_x[train_mask], s_c[train_mask]
    shadow_test_performance = s_x[np.bitwise_not(train_mask)], s_c[np.bitwise_not(train_mask)]
    train_mask = l == 1
    target_train_performance = pred_score[train_mask], true_label[train_mask]
    target_test_performance = pred_score[np.bitwise_not(train_mask)], true_label[np.bitwise_not(train_mask)]
    acc = song_membership_inference(shadow_train_performance, shadow_test_performance,
                                    target_train_performance, target_test_performance,
                                    len(np.unique(true_label)))
    print(f"song MIR: {acc}")

    # Ye 成员推理
    s_train, s_test = get_data_set(exp_config, DS, data_set, model_name, shadow=True)
    s_train_loader = DataLoader(s_train, batch_size=batch_size)
    s_test_loader = DataLoader(s_test, batch_size=batch_size)
    acc, auc = Ye_membership_inference_attack(model, shadow_model, train_loader, test_loader,
                                              s_train_loader,
                                              s_test_loader)
    print(f"Ye MIR: {acc}  , auc: {auc}")
