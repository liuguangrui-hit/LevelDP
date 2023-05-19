from torch.utils.data import DataLoader

from core.Experiments import config_dict
from core.Models import load_model
from core.Models import train as trainer
from util.DataSet import *

delta_f = None

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:{device}")
    for key in config_dict.keys():  # key是数据集的名字
        print(f"dataset: {key}")
        config = config_dict.get(key)
        DS = config["data_set"]
        shadow_num = config["shadow_num"]
        exps = config["exp"]
        for exp in exps.keys():  # exp是模型的名字
            print(f"model: {exp}")
            exp_config = exps[exp]
            # 加载实验参数
            batch_size = exp_config["bs"]
            lr = exp_config["lr"]
            epochs = exp_config["epoch"]
            model_proto = exp_config["model"]
            t_model_path = f"state_dicts/{key}_{exp}_target.pt"
            s_model_path = f"state_dicts/{key}_{exp}_shadow.pt"
            train, test = get_data_set(exp_config, DS, key, exp)

            train_loader = DataLoader(train, batch_size=batch_size)
            test_loader = DataLoader(test, batch_size=batch_size)
            if exp_config.get("in_size") is None:
                model = model_proto(classes_num=config.get("class_num")).to(device)
            else:
                model = model_proto(classes_num=config.get("class_num"), in_size=exp_config.get("in_size")).to(
                    device)

            if os.path.exists(t_model_path):
                model = load_model(model, t_model_path)
            else:
                model = trainer(model, (train_loader, test_loader), lr, epochs,
                                save_path=t_model_path,
                                visible=False, device=device)

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
            train_acc = np.sum(np.argmax(pred_score, axis=1)[l == 1] == true_label[l == 1]) / len(true_label[l == 1])
            print(f"test acc: {acc}")
