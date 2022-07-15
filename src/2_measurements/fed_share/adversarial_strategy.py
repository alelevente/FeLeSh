from functools import total_ordering
from pydoc import cli
import flwr as fl
import sys
import pandas as pd

sys.path.append("../..")
import net
import tools

from collections import OrderedDict
import torch
import torch.nn.functional as F

from flwr.server.strategy.aggregate import aggregate

from flwr.common import (
    parameters_to_weights,
    weights_to_parameters,
)

import pandas as pd

MNIST_DIGITS_PATH = "../../../data/MNIST/digits/"
MNIST_COMPLETE_PATH = "../../../data/MNIST/mnist_train.csv"
RESULT_PATH = "../../../results/fed_fixshare/"
FIXED_SHARING_PATH = "../../..data/participants/fix_shared/shared.csv"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AdversarialStrategyFix(fl.server.strategy.FedAvg):

    def __init__(self, n_client, **kwargs):
        super().__init__(**kwargs)
        #creating the representation of the clients =: "shadow clients":
        self.clients = []     
        self.global_model = net.Net().to(DEVICE)
        self.client_net = net.Net().to(DEVICE)
        self.client_map = {}
        self.client_names = []
        self.used_clients = 0

        self.digit_data = [tools.create_data_loaders(
            df_path = MNIST_DIGITS_PATH + "digit%d.csv"%i) for i in range(10)]
        _, self.test_data = tools.create_data_loaders(df_path = MNIST_COMPLETE_PATH, test_portion=0.33)
        self.data_to_share = pd.read_csv(FIXED_SHARING_PATH)
        self.adversary_save_file = open(RESULT_PATH+"adv_results.csv", "w")
        self.adversary_save_file.write("name,round,digit,accuracy\n")
        self.global_save_file = open(RESULT_PATH+"global.csv", "w")
        self.global_save_file.write("round,accuracy,loss\n")
        
    def __del__(self):
        self.adversary_save_file.close()
        self.global_save_file.close()

    def _infer(self, client_net):
        """Computes per digit accuracies for a client"""
        accuracies = []
        with torch.no_grad():
            for digit in range(10):
                total = 0
                correct = 0
                #accessing the test data loader:
                test_loader = self.digit_data[digit][1]
                for d in test_loader:
                    images, labels = d[0].to(DEVICE), d[1].to(DEVICE)
                    outputs = client_net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                accuracies.append(correct/total)

        if torch.cuda.is_available:
            torch.cuda.empty_cache()
        return accuracies

    def _test_global(self):
        """Computes global model accuracy and loss"""
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for d in self.test_data:
                images, labels = d[0].to(DEVICE), d[1].to(DEVICE)
                outputs = self.global_model(images)
                loss += F.nll_loss(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        if torch.cuda.is_available:
            torch.cuda.empty_cache()
        accuracy = correct/total
        return loss/total, accuracy
    
    def configure_fit(
        self, rnd: int, parameters, client_manager
    ):
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)
        config["round"] = rnd
        config["shared_data"] = self.data_to_share
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        rnd,
        results,
        failures,
    ):
        """Aggregate fit results using weighted average.
            Also, performs an adversarial evaluation."""
        
        #updating shadow clients:
        for client, fit_res in results:
            #transforming client objects to indices:
            if not(client in self.client_map):
                self.client_map[client] = self.used_clients
                self.client_names.append(fit_res.metrics["name"])
                self.clients.append(None)
                self.used_clients += 1
            c = self.client_map[client]

            parameters = parameters_to_weights(fit_res.parameters)
            params_dict = zip(self.client_net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.clients[c] = state_dict

        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]
        agg_weights = aggregate(weights_results)
        params_dict = zip(self.global_model.state_dict().keys(), agg_weights)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        #updating the global model:
        self.global_model.load_state_dict(state_dict, strict=True)

        #infer data distribution:
        self.model_accuracies = []
        for x in range(self.used_clients):

            self.client_net.load_state_dict(self.clients[x], strict=True)
            self.model_accuracies.append(self._infer(client_net=self.client_net))
        self.global_accuracy = self._infer(self.global_model)
        for i in range(self.used_clients):
            print(self.client_names[i] + "\taccuracy:\t", self.model_accuracies[i])
        print("global\taccuracy:\t", self.global_accuracy)

        #saving results for further analysis:
        #client data:
        for i in range(self.used_clients):
            name = self.client_names[i]
            for digit in range(10):
                self.adversary_save_file.write("%s,%d,%d,%f\n"%(name,
                                                               rnd,
                                                               digit,
                                                               self.model_accuracies[i][digit]))
        #global model:
        for digit in range(10):
            self.adversary_save_file.write("global,%d,%d,%f\n"%(rnd, digit, self.global_accuracy[digit]))

        #saving global model results:
        loss, accuracy = self._test_global()
        self.global_save_file.write("%d,%f,%f\n"%(rnd, accuracy, loss))
        
        return super().aggregate_fit(rnd, results, failures)
        
    
