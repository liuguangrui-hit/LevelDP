import argparse

from torch.utils.data import DataLoader

from core.Attack import apgd_with_data, NIFGSM
from core.Models import load_model, MLP
from util.DataSet import *
from util.DataSet import get_data_set

delta_f = None


def add_noise(data, label, noise, noise_feature):
    if len(noise_feature[0]) == 0:
        return data
    shape = data.shape
    data = data.reshape(shape[0], -1)
    noise = noise.reshape(shape[0], -1)
    noise = noise.to(data.device)
    for l in range(10):
        mask = label == l
        d = data[mask]
        n = noise[mask]
        d[:, noise_feature[l]] += n[:, noise_feature[l]]
        data[mask] = d
    data = data.reshape(shape)
    return data


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
                    "bs": 500,
                    "in_size": 784,
                },
            }
        }
    }

    parser = argparse.ArgumentParser(description='Add noise on feature one by one')
    parser.add_argument('--noise_order', type=str, default="12")
    parser.add_argument('--epsilon', type=float, default=1e-1)
    args = parser.parse_args()
    noise_order = args.noise_order
    epsilon = args.epsilon

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for key in config_dict.keys():  # key是数据集的名字
        print(f"dataset: {key}")
        config = config_dict.get(key)
        DS = config["data_set"]
        shadow_num = config["shadow_num"]
        exps = config["exp"]
        class_num = config.get("class_num")
        if class_num is None:
            class_num = 10

        feature_level = np.load(f"attention/features/{key}.npy", allow_pickle=True)

        feature_1_2 = feature_level[0][0] + feature_level[1][0]
        feature_3_4 = feature_level[2][0] + feature_level[3][0]

        # 加载重要特征
        attentions = None
        feature_rank = None

        for i in range(class_num):
            a = np.load(f"attention/attention/{key}/{i}.npy")

            if attentions is None:
                attentions = a
            else:
                attentions = np.row_stack((attentions, a))

        feature_rank = []
        if noise_order == "random":
            feature_rank = None
            for i in range(len(attentions)):
                a = np.asarray(range(28 * 28))
                np.random.shuffle(a)
                a = a.reshape((1, -1))
                if feature_rank is None:
                    feature_rank = a
                else:
                    feature_rank = np.row_stack((feature_rank, a))
        else:
            for i in range(len(attentions)):
                if noise_order == "12":
                    tmp = feature_1_2.copy()
                    tmp.sort(key=lambda x: attentions[i][x], reverse=True)
                    tmp_ = feature_3_4.copy()
                    tmp_.sort(key=lambda x: attentions[i][x], reverse=True)
                elif noise_order == "34":
                    tmp = feature_3_4.copy()
                    tmp.sort(key=lambda x: attentions[i][x], reverse=False)
                    tmp_ = feature_1_2.copy()
                    tmp_.sort(key=lambda x: attentions[i][x], reverse=False)
                feature_rank.append(tmp + tmp_)
        feature_rank = np.asarray(feature_rank)

        delta_f = np.max(np.load(f"delta_f/{key}.npy")[4])
        b = delta_f / epsilon


        for exp in exps.keys():
            print(f"model: {exp}")
            exp_config = exps[exp]
            # 加载实验参数
            batch_size = exp_config["bs"]
            lr = exp_config["lr"]
            epochs = exp_config["epoch"]
            model_proto = exp_config["model"]
            if exp_config.get("in_size") is None:
                model = model_proto(classes_num=config.get("class_num")).to(device)
            else:
                model = model_proto(classes_num=config.get("class_num"), in_size=exp_config.get("in_size")).to(
                    device)

            model = load_model(model, f"state_dicts/{key}_{exp}_target.pt")

            _, test = get_data_set(exp_config, DS, "mnist", "MLP")
            test_loader = DataLoader(test, batch_size=batch_size)

            # 噪声大小和位置处理
            m = torch.distributions.Laplace(0, b)

            ans = torch.zeros((3, 785))
            atk = NIFGSM(model, steps=5, eps=8 / 255)

            for feature_num in range(0, 785):
                noise_feature = feature_rank[:, :feature_num]
                nifgsm_cnt = 0
                apgd_cnt = 0
                cnt = 0
                corr_cnt = 0
                for data, label in test_loader:
                    data, label = data.to(device), label.to(device)
                    noise = m.sample(data.shape).to(device)
                    test_data = add_noise(data.clone(), label, noise, noise_feature)
                    out = torch.argmax(model(test_data), dim=1)
                    mask = (out == label).detach().cpu()
                    corr_cnt += torch.sum(mask)
                    cnt += len(mask)

                    noise = m.sample(data.shape)
                    apgd_data = apgd_with_data(model, data, label, class_num=config.get("class_num"))
                    test_data = add_noise(apgd_data.clone(), label, noise, noise_feature)
                    out = torch.argmax(model(test_data), dim=1)
                    mask = (out == label).detach().cpu()
                    apgd_cnt += torch.sum(mask)

                    noise = m.sample(data.shape)
                    nifgsm_data = atk(data, label)
                    test_data = add_noise(nifgsm_data.clone(), label, noise, noise_feature)
                    out = torch.argmax(model(test_data), dim=1)
                    mask = (out == label).detach().cpu()
                    nifgsm_cnt += torch.sum(mask)

                ans[0][feature_num] = corr_cnt / cnt
                ans[1][feature_num] = 1 - (nifgsm_cnt / cnt)
                ans[2][feature_num] = 1 - (apgd_cnt / cnt)

                np.save(f"state_dicts/noise_by_feature_Adver/mnist_MLP_{noise_order}.npy", ans)
                np.savetxt(f"state_dicts/noise_by_feature_Adver/mnist_MLP_{noise_order}.csv",
                           ans.detach().cpu().numpy(), delimiter=",", encoding="utf-8")
    print(f"results saved at state_dicts/noise_by_feature_Adver/")
