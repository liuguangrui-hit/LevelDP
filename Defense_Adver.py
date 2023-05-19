import argparse

from torch.utils.data import DataLoader

from core.Attack import apgd_with_data, NIFGSM
from core.Experiments import config_dict
from core.Models import load_model
from util.DataSet import *
from util.DataSet import get_data_set

delta_f = None


def add_noise(data, noise, noise_feature):
    if len(noise_feature) == 0:
        return data
    shape = data.shape
    data = data.reshape(shape[0], -1)
    noise = noise.reshape(shape[0], -1)
    noise = noise.to(data.device)
    data[:, noise_feature] += noise[:, noise_feature]
    data = data.reshape(shape)
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Add noise on feature one by one')
    parser.add_argument('--epsilon', type=float, default=1e-1)
    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument('--model', type=str, default="MLP")
    args = parser.parse_args()
    epsilon = args.epsilon
    data_set = args.dataset
    model_name = args.model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = config_dict.get(data_set)
    DS = config["data_set"]
    shadow_num = config["shadow_num"]
    exps = config['exp']
    class_num = config.get("class_num")
    if class_num is None:
        class_num = 10

    feature_level = np.load(f"attention/features/{data_set}.npy", allow_pickle=True)
    feature_1_2 = feature_level[0][0] + feature_level[1][0]

    delta_f = np.max(np.load(f"delta_f/{data_set}.npy")[4])
    b = delta_f / epsilon

    exp_config = exps[model_name]
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

    model = load_model(model, f"state_dicts/{data_set}_{model_name}_target.pt")

    _, test = get_data_set(exp_config, DS, data_set, model_name)
    test_loader = DataLoader(test, batch_size=batch_size)

    m = torch.distributions.Laplace(0, b)

    ans = torch.zeros((3, 785))
    atk = NIFGSM(model, steps=5, eps=8 / 255)

    noise_feature = feature_1_2  # 确定加噪声的范围
    # 测防御模型干净样本准确率
    nifgsm_cnt = 0
    apgd_cnt = 0
    cnt = 0
    corr_cnt = 0
    for data, label in test_loader:
        data, label = data.to(device),label.to(device)
        noise = m.sample(data.shape)
        test_data = add_noise(data.clone(), noise, noise_feature).to(device)
        out = torch.argmax(model(test_data), dim=1)
        mask = (out == label).detach().cpu()
        corr_cnt += torch.sum(mask)
        cnt += len(mask)

        noise = m.sample(data.shape)
        apgd_data = apgd_with_data(model, data, label, class_num=config.get("class_num"))
        test_data = add_noise(apgd_data.clone(), noise, noise_feature).to(device)
        out = torch.argmax(model(test_data), dim=1)
        mask = (out == label).detach().cpu()
        apgd_cnt += torch.sum(mask)

        noise = m.sample(data.shape)
        nifgsm_data = atk(data, label)
        test_data = add_noise(nifgsm_data.clone(), noise, noise_feature).to(device)
        out = torch.argmax(model(test_data), dim=1)
        mask = (out == label).detach().cpu()
        apgd_cnt += torch.sum(mask)

    print(f"acc: {corr_cnt / cnt}")
    print(f"nifgsm AER: {1 - (nifgsm_cnt / cnt)}")
    print(f"apgd AER: {1 - (apgd_cnt / cnt)}")
