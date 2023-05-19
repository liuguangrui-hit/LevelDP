from torch.utils.data import DataLoader

from core.Attack import NIFGSM, apgd_with_data
from core.Experiments import config_dict
from core.Models import load_model
from util.DataSet import *

delta_f = None

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:{device}")
    for key in config_dict.keys():
        print(f"dataset: {key}")
        config = config_dict.get(key)
        DS = config["data_set"]
        shadow_num = config["shadow_num"]
        exps = config["exp"]
        for exp in exps.keys():
            print(f"model: {exp}")
            exp_config = exps[exp]
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

            model = load_model(model, t_model_path)

            atk = NIFGSM(model, steps=5, eps=8 / 255)
            all_cnt = 0
            adver_cnt = 0
            for data, label in test_loader:
                data, label = data.to(device), label.to(device)
                adver_data = atk(data, label)
                out = torch.argmax(model(adver_data), dim=1)
                all_cnt += len(out)
                adver_cnt += torch.sum(out == label)
            print(f"nifgsm bypass rate = {1 - adver_cnt / all_cnt}")

            # apgd
            all_cnt = 0
            adver_cnt = 0
            for data, label in test_loader:
                data, label = data.to(device), label.to(device)
                adver_data = apgd_with_data(model, data, label, class_num=config.get("class_num"))
                out = torch.argmax(model(adver_data), dim=1)
                all_cnt += len(out)
                adver_cnt += torch.sum(out == label)
            print(f"apgd bypass rate  = {1 - adver_cnt / all_cnt}")
