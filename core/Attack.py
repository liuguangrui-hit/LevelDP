import numpy as np
import torch
import torch.nn as nn
from core.APGD import AutoProjectedGradientDescent as APGD
from art.estimators.classification import PyTorchClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchattacks.attack import Attack

from core.Mentr import black_box_benchmarks
from core.Models import train as trainer, LSTM, MLP
from util.DataSet import CICIDS_2017

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fix_label(model, name, save_path=None, dataset=None, exp_config=None, data_folder=None, npy=False, memguard=False):
    if npy is None:
        npy = False
    if dataset is None:
        raise NotImplementedError()
    else:
        if exp_config.get("transform") is not None:
            train = dataset()
            if isinstance(train, CICIDS_2017) and isinstance(model, LSTM):
                ds = dataset(kind=name, min=train.mean, max=train.max, transform=exp_config["transform"],
                             unsqueeze=True)
            else:
                ds = dataset(kind=name, min=train.min, max=train.max, transform=exp_config["transform"])
        else:
            train = dataset()
            if isinstance(train, CICIDS_2017) and isinstance(model, LSTM):
                ds = dataset(kind=name, min=train.mean, max=train.max, unsqueeze=True)
            else:
                ds = dataset(kind=name, min=train.min, max=train.max)
    batch_size = exp_config["bs"]
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        # 分别修改标签
        pl = []
        for d, l in loader:
            d = d.to(device)
            if memguard:
                model.memlabels = l
            o = model(d)
            o = torch.softmax(o, dim=1)
            pred_l = torch.argmax(o, dim=1)
            pl.append(pred_l.cpu().numpy())
        pl = np.concatenate(pl)
        if memguard:
            model.memlabels = None
        if data_folder is None:
            raise NotImplementedError()
        if npy:
            fixed = np.load(f"{data_folder}{name}.npy")
            fixed = fixed[:, :-1]
            if save_path is not None:
                # np.savetxt(save_path, fixed, "%s", delimiter=",", encoding="UTF-8")
                np.save(save_path + ".npy", np.column_stack((fixed, pl)))
            else:
                np.save(f"{data_folder}{name}.npy", np.column_stack((fixed, pl)))
        else:
            fixed = torch.load(f"{data_folder}{name}.pt")
            if save_path is not None:
                # np.savetxt(save_path, fixed, "%s", delimiter=",", encoding="UTF-8")
                torch.save((fixed[0], torch.tensor(pl)), save_path + ".pt")
            else:
                torch.save((fixed[0], torch.tensor(pl)), f"{data_folder}{name}.pt", )


def extract_data(data_loader):
    x, y = [], []
    for data, label in data_loader:
        x.append(data.cuda())
        y.append(label.cuda())
    x = torch.row_stack(x)
    y = torch.concat(y)
    return x, y


def Ye_membership_inference_attack(target, shadow, train_loader, test_loader, s_train_loader, s_test_loader):
    from privacy_meter.audit import Audit
    from privacy_meter.constants import InferenceGame
    from privacy_meter.dataset import Dataset
    from privacy_meter.hypothesis_test import threshold_func
    from privacy_meter.information_source import InformationSource
    from privacy_meter.information_source_signal import ModelLoss
    from privacy_meter.metric import ShadowMetric
    from privacy_meter.model import PytorchModel
    criterion = nn.CrossEntropyLoss(reduction='sum')
    target_model = PytorchModel(
        model_obj=target,
        loss_fn=criterion
    )
    shadow_model = PytorchModel(
        model_obj=shadow,
        loss_fn=criterion
    )
    x_train, y_train = extract_data(train_loader)
    x_test, y_test = extract_data(test_loader)
    s_x_train, s_y_train = extract_data(s_train_loader)
    s_x_test, s_y_test = extract_data(s_test_loader)
    del train_loader
    del test_loader
    dataset1 = Dataset(
        data_dict={'train': {'x': x_train, 'y': y_train}, 'test': {'x': x_test, 'y': y_test}},
        default_input='x',
        default_output='y'
    )
    dataset2 = Dataset(
        data_dict={'train': {'x': s_x_train, 'y': s_y_train}, 'test': {'x': s_x_test, 'y': s_y_test}},
        default_input='x',
        default_output='y'
    )
    datasets_list = [dataset1, dataset2]
    target_info_source = InformationSource(
        models=[target_model],
        datasets=[datasets_list[0]]
    )
    reference_info_source = InformationSource(
        models=[shadow_model],
        datasets=datasets_list[1:]
    )

    metric = ShadowMetric(
        target_info_source=target_info_source,
        reference_info_source=reference_info_source,
        signals=[ModelLoss()],
        hypothesis_test_func=threshold_func,
        unique_dataset=False,
        reweight_samples=True
    )
    audit = Audit(
        inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
        metrics=metric,
        target_info_sources=target_info_source,
        reference_info_sources=reference_info_source
    )
    audit.prepare()
    result = audit.run()[0]
    return result.accuracy, result.roc_auc


def song_membership_inference(shadow_train_performance, shadow_test_performance, target_train_performance,
                              target_test_performance, class_num):
    bb = black_box_benchmarks(shadow_train_performance, shadow_test_performance, target_train_performance,
                              target_test_performance, class_num)
    return bb._mem_inf_benchmarks(all_methods=False, benchmark_methods=['modified entropy'])


def train_single_shadow_model(index, dataset=None, exp_config=None, return_model=False, classes_num=None,
                              model_path=None):
    if dataset == CICIDS_2017:
        classes_num = 7
    x = []
    y = []
    classes = []
    cpu = torch.device("cpu")
    if exp_config is None:
        raise NotImplementedError()
    else:
        model_proto = exp_config.get("model")
        if exp_config.get("in_size") is None:
            model = model_proto(classes_num=classes_num).to(device)
        else:
            if dataset == CICIDS_2017:
                model = MLP(in_size=78, classes_num=classes_num).to(device)
            else:
                model = model_proto(in_size=exp_config.get("in_size"), classes_num=classes_num).to(device)
    if dataset is None:
        raise NotImplementedError()
    else:
        if exp_config.get("transform") is not None:
            if dataset == CICIDS_2017 and isinstance(model, LSTM):
                train = dataset(f"shadow_train_fixed_{index}", transform=exp_config["transform"], unsqueeze=True)
                test = dataset(f"shadow_test_fixed_{index}", min=train.min, max=train.max,
                               transform=exp_config["transform"], unsqueeze=True)
            else:
                train = dataset(f"shadow_train_fixed_{index}", transform=exp_config["transform"])
                test = dataset(f"shadow_test_fixed_{index}", min=train.min, max=train.max,
                               transform=exp_config["transform"])
        else:
            if dataset == CICIDS_2017 and isinstance(model, LSTM):
                train = dataset(f"shadow_train_fixed_{index}", unsqueeze=True)
                test = dataset(f"shadow_test_fixed_{index}", min=train.min, max=train.max, unsqueeze=True)
            else:
                train = dataset(f"shadow_train_fixed_{index}")
                test = dataset(f"shadow_test_fixed_{index}", min=train.min, max=train.max)
                # train = dataset(f"shadow_train_{index}")
                # test = dataset(f"shadow_test_{index}", min=train.min, max=train.max)
    if exp_config is None:
        raise NotImplementedError()
    else:
        train_loader = DataLoader(train, batch_size=exp_config["bs"], shuffle=True)
        test_loader = DataLoader(test, batch_size=exp_config["bs"])
        if model_path is not None:
            model = trainer(model, train_loader, lr=exp_config["lr"],
                            epochs=exp_config["epoch"],
                            save_path=model_path,
                            device=device, visible=False)
        else:
            model = trainer(model, train_loader, lr=exp_config["lr"],
                            epochs=exp_config["epoch"],
                            save_path=None,
                            device=device, visible=False)
    with torch.no_grad():
        for data, label in train_loader:
            data = data.to(device)
            out = torch.softmax(model(data), 1)
            x += out.to(cpu).tolist()
            y += [1] * len(out)
            classes += label.tolist()
        pred_label = []
        true_label = []
        for data, label in test_loader:
            data = data.to(device)
            out = torch.softmax(model(data), 1)
            x += out.to(cpu).tolist()
            y += [0] * len(out)
            classes += label.tolist()
            true_label += label.tolist()
            pred_label += torch.argmax(out, 1).tolist()
    print("------------------------------------------")
    print(f"shadow_{index}:")
    print(accuracy_score(true_label, pred_label))
    # a, b = metrics_report(true_label, pred_label)
    # print(a)
    # print(b)
    print("------------------------------------------")
    if return_model:
        return x, y, classes, model
    return x, y, classes


def apgd_with_data(model, data, label, class_num=None, max_iter=1, eps=8/255, eps_step=0.1, batch_size=512):
    if class_num is None:
        class_num = 10
    classifier = PyTorchClassifier(model, loss=nn.CrossEntropyLoss(), input_shape=data[0].shape,
                                   nb_classes=class_num)
    attack = APGD(classifier, verbose=False, batch_size=batch_size, eps=eps, eps_step=eps_step, max_iter=max_iter)
    data, label = data.to(device), label.to(device)
    data = torch.tensor(attack.generate(data.cpu().detach().numpy(), label.cpu().detach().numpy()))
    data = data.to(device)
    return data


class NIFGSM(Attack):
    r"""
        reference torchattacks
    """

    def __init__(self, model, eps=0.3, alpha=2 / 255, steps=20, decay=1):
        super().__init__("NIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        self.model.train()
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            nes_images = adv_images + self.decay * self.alpha * momentum
            outputs = self.get_logits(nes_images)
            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            num_dims = len(images.shape)
            mean_dims = tuple([iiidx + 1 for iiidx in range(num_dims - 1)])
            grad = self.decay * momentum + grad / torch.mean(torch.abs(grad), dim=mean_dims, keepdim=True)
            momentum = grad
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        return adv_images

#
