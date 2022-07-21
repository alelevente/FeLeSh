from collections import OrderedDict
from http import client
import pandas as pd

import torch
import torch.nn.functional as F

import flwr as fl

import sys
sys.path.append("../..")

import net
import tools

RESULT_PATH = "../../../results/fed_iid/"

def train(net, train_loader, epochs, device):
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=.9)
    losses = []
    for _ in range(epochs):
        l_, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = F.nll_loss(net(images), labels)
            l_ += loss.item()
            total += labels.size(0)
            loss.backward()
            optimizer.step()
        losses.append(l_)
    return losses

def test(net, test_loader, device):
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += F.nll_loss(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return loss, accuracy

class MnistClient(fl.client.NumPyClient):
    def __init__(self, name, path, cuda, **kwargs):
        super().__init__()
        self.name = name
        self.data_path = path
        self.train_set, self.test_set = tools.create_data_loaders(path)
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and cuda) else "cpu")
        self.net = net.Net().to(self.device)
        self.result_file_path = RESULT_PATH+name+".csv"
        self.train_round = 1
        self.test_round = 1     
        self.data = None

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        losses = train(self.net, self.train_set, epochs=1, device=self.device)
        results = pd.read_csv(self.result_file_path) if self.train_round > 1 else pd.DataFrame()
        round_ = self.train_round
        if "round" in config:
            round_ = config["round"]
        for i, l in enumerate(losses):
            results = results.append({
                "metric": "train_loss",
                "value": l*len(self.test_set)/len(self.train_set),
                "round": round_,
                "epoch": i}, ignore_index = True)
        results.to_csv(self.result_file_path, index=False)
        self.train_round += 1
        return self.get_parameters(), len(self.train_set), {"name": self.name}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.test_set, self.device)
        if not((self.test_round == 1) and (self.train_round == 1)):
            results = pd.read_csv(self.result_file_path)
        else:
            results = pd.DataFrame()
        round_ = self.test_round
        if "round" in config:
            round_ = config["round"]
        results = results.append({
            "metric": "test_loss",
            "value": loss,
            "round": round_,
            "epoch": -1
        }, ignore_index=True)
        results = results.append({
            "metric": "test_accuracy",
            "value": accuracy,
            "round": round_,
            "epoch": -1
        }, ignore_index=True)
        results.to_csv(self.result_file_path, index=False)
        self.test_round += 1
        return float(loss), len(self.test_set), {"accuracy": float(accuracy)}
