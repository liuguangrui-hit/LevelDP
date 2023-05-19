from torch.utils.data import DataLoader

from core.Attack import fix_label, \
    train_single_shadow_model, song_membership_inference, Ye_membership_inference_attack
from core.Experiments import config_dict
from core.Models import load_model
from util.DataSet import *

delta_f = None

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:{device}")
    for key in config_dict.keys():  # key是数据集的名字
        if key == "cicids":
            continue
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

            model = load_model(model, t_model_path)

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
                                                                    model_path=s_model_path)

            #   成员推理
            # song成员推理
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
            print(f"song membership inference acc : {acc}")

            # Ye 成员推理
            s_train, s_test = get_data_set(exp_config, DS, key, exp, shadow=True)
            s_train_loader = DataLoader(s_train, batch_size=batch_size)
            s_test_loader = DataLoader(s_test, batch_size=batch_size)
            acc, auc = Ye_membership_inference_attack(model, shadow_model, train_loader, test_loader, s_train_loader,
                                                      s_test_loader)
            print(f"Ye membership inference acc : {acc}  , auc: {auc}")
