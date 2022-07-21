from collections import OrderedDict
from http import client
import pandas as pd
import time

import torch
import torch.nn.functional as F

import flwr as fl

import sys
sys.path.append("../..")

import net
import tools

RESULT_PATH = "../../../results/fed_changeshare/"

def wait_for_cuda():
    while not(tools.is_mem_enough()):
        time.sleep(0.1)

def train(net, train_loader, epochs, device):
    #wait_for_cuda()
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
    #del labels
    #del images
    #torch.cuda.empty_cache()
    return losses

def test(net, test_loader, device):
    #wait_for_cuda()
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
    #torch.cuda.empty_cache()
    return loss, accuracy

class MnistClientShare(fl.client.NumPyClient):
    def __init__(self, name, path, cuda, **kwargs):
        super().__init__()
        self.name = name
        self.data_path = path
        self.own_data = pd.read_csv(path)
        self.train_set, self.test_set = tools.create_data_loaders(df = self.own_data)
        self.device = torch.device("cuda:0" if cuda and torch.cuda.is_available() else "cpu")
        #torch.cuda.empty_cache()
        self.net = net.Net().to(self.device)
        self.result_file_path = RESULT_PATH+name+".csv"
        self.train_round = 1
        self.test_round = 1     
        self.data = None
        self.save_file = open(self.result_file_path, "w")
        self.save_file.write("metric,value,round,epoch\n")
        
    def __del__(self):
        self.save_file.close()
        
    def _merge_sample_data(self, shared):
        """Merges shared data with own data. Returned DataFrame contains evenly sampled data"""
        merged_data = pd.concat([self.own_data, shared]).reset_index(drop=True)
        min_label_count = min(merged_data["label"].value_counts())
        new_dataset = None
        for digit in range(10):
            new_dataset = pd.concat([new_dataset, merged_data[merged_data["label"] == digit].sample(n=min_label_count)])
        size_diff = len(self.own_data) - len(new_dataset)
        if size_diff > 0:
            new_dataset = pd.concat([new_dataset, self.own_data.sample(n=size_diff)])
        return new_dataset
        

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        if "reload" in config:
            received_data = pd.read_csv(config["reload"])
            combined_data = self._merge_sample_data(received_data)
            print("received:", len(received_data), "combined: ", len(combined_data))
            self.train_set, self.test_set = tools.create_data_loaders(df=combined_data)
        
        losses = train(self.net, self.train_set, epochs=1, device=self.device)
        round_ = self.train_round
        if "round" in config:
            round_ = config["round"]

        for i, l in enumerate(losses):
            self.save_file.write("train_loss,%f,%d,%d\n"%(l*len(self.test_set)/len(self.train_set),
                                                         round_,
                                                         i))
        self.train_round += 1
        return self.get_parameters(), len(self.train_set), {"name": self.name}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.test_set, self.device)
        round_ = self.test_round
        if "round" in config:
            round_ = config["round"]
        self.save_file.write("test_loss,%f,%d,-1\n"%(loss, round_))
        self.save_file.write("test_accuracy,%f,%d,-1\n"%(accuracy, round_))
        self.test_round += 1
        return float(loss), len(self.test_set), {"accuracy": float(accuracy)}
