# 测试四级特征分级加噪

import argparse

from torch.utils.data import DataLoader

from ExtractFeature import ExtractFeature
from core.Attack import song_membership_inference, fix_label, \
    train_single_shadow_model, Ye_membership_inference_attack
from core.Models import MLP
from core.Models import train as trainer
from util.DataSet import *

random_perturbation = None
for i in range(10):
    # 随机重要性
    a = np.asarray(range(28 * 28))
    np.random.shuffle(a)
    a = a.reshape((1, -1))
    if random_perturbation is None:
        random_perturbation = a
    else:
        random_perturbation = np.row_stack((random_perturbation, a))


def get_data(dataset, data_path):
    if dataset == "cicids":
        data = np.load(os.path.join(data_path, "train.npy"))
        label = torch.tensor(data[:, -1])
        data = torch.tensor(data[:, :-1])
        dataset = CICIDS_2017()
        ma = dataset.max
        mi = dataset.min
        data = (data - mi) / (ma - mi + 1e-6)
    else:
        data, label = torch.load(os.path.join(data_path, "train.pt"))
        data, label = torch.tensor(data) / 255, torch.tensor(label)
    shape = data.shape
    if dataset == "cifar":
        data = data.reshape((shape[0], -1, 3))
    else:
        data = data.reshape((shape[0], -1))
    return data, label, shape


def get_feature_rank(dataset, order="23"):
    ef = ExtractFeature()
    attentions = ef.load_attentions(dataset)
    f_num = attentions.shape[1]
    f = np.load(f"attention/features/{dataset}.npy", allow_pickle=True)  # 4*classes
    feature_rank = []
    for i in range(len(attentions)):
        if order == "23":
            tmp = f[1][i] + f[2][i]
            tmp.sort(key=lambda x: attentions[i][x], reverse=True)
            tmp_ = list(set(list(range(f_num))) - set(tmp))
            tmp_.sort(key=lambda x: attentions[i][x], reverse=True)
        elif order == "14":
            tmp = f[0][i] + f[3][i]
            tmp.sort(key=lambda x: attentions[i][x], reverse=False)
            tmp_ = list(set(list(range(f_num))) - set(tmp))
            tmp_.sort(key=lambda x: attentions[i][x], reverse=False)
        feature_rank.append(tmp + tmp_)
    return feature_rank


def add_noise(ep, dataset, data_path, order="23", noise_feature_num=0):
    if order == "random":
        feature_rank = random_perturbation[:, :noise_feature_num]
    else:
        feature_rank = np.asarray(get_feature_rank(dataset, order))[:, :noise_feature_num]
    b = np.load(f"delta_f/{dataset}.npy")[5] / ep
    data, label, shape = get_data(dataset, data_path)
    if noise_feature_num != 0:
        for i in range(len(b)):
            mask = label == i
            d = data[mask]  # 筛选某一类别特征
            num = len(d)
            m = torch.distributions.Laplace(0, torch.tensor([b[i]]))
            noise = m.sample((num, len(feature_rank[i]))).squeeze(-1)  # 噪声采样
            d[:, feature_rank[i]] += noise
            data[mask] = d
    data = data.reshape(shape)
    return data, label


# -------------------------
if __name__ == "__main__":
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
                    "bs": 5,
                    "in_size": 784,
                    "feature_num": 784
                },
            }
        },
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:{device}")

    parser = argparse.ArgumentParser(description='Add noise on feature one by one')
    parser.add_argument('--noise_order', type=str, default="23")
    parser.add_argument('--epsilon', type=float, default=1e-3)
    args = parser.parse_args()
    noise_order = args.noise_order
    ep = args.epsilon

    config = config_dict.get("mnist")
    DS = config["data_set"]
    shadow_num = config["shadow_num"]
    exps = config["exp"]
    exp_config = exps["MLP"]
    batch_size = exp_config["bs"]
    lr = exp_config["lr"]
    epochs = exp_config["epoch"]
    model_proto = exp_config["model"]
    feature_num = exp_config["feature_num"]
    dots = np.zeros((5, feature_num + 1))
    # 加载实验参数
    ls = list(range(0, 785, 10))
    for idx in ls:
        train_data, train_label = add_noise(ep, "mnist", data_path=config["data_path"],
                                            order=noise_order, noise_feature_num=idx)
        if isinstance(train_data, torch.Tensor):
            train_data = train_data.numpy()
        if isinstance(train_label, torch.Tensor):
            train_label = train_label.numpy()
        train, test = get_data_set(exp_config, DS, "mnist", "MLP", t_data=train_data, t_label=train_label)
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
        train, _ = get_data_set(exp_config, DS, "mnist", "MLP")
        train_loader = DataLoader(train, batch_size=batch_size)
        # 测试目标模型准确率
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
        dots[0, idx] = acc
        # print(f"target acc : {acc}")

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
        target_test_performance = pred_score[np.bitwise_not(train_mask)], true_label[
            np.bitwise_not(train_mask)]
        acc = song_membership_inference(shadow_train_performance, shadow_test_performance,
                                        target_train_performance, target_test_performance,
                                        len(np.unique(true_label)))
        dots[1, idx] = acc
        # print(f"song membership inference acc : {acc}")

        s_train, s_test = get_data_set(exp_config, DS, "mnist", "MLP", shadow=True)
        s_train_loader = DataLoader(s_train, batch_size=batch_size)
        s_test_loader = DataLoader(s_test, batch_size=batch_size)
        acc, auc = Ye_membership_inference_attack(model, shadow_model, train_loader, test_loader,
                                                  s_train_loader,
                                                  s_test_loader)
        dots[2, idx] = acc
        dots[3, idx] = auc

        # print(f"Ye membership inference acc : {acc}")
        np.save(f"state_dicts/noise_by_feature_MIA/mnist_MLP_{noise_order}.npy", dots)
        np.savetxt(f"state_dicts/noise_by_feature_MIA/mnist_MLP_{noise_order}.csv", dots, delimiter=",",
                   encoding="utf-8")

    print(f"results saved at state_dicts/noise_by_feature_MIA/")
