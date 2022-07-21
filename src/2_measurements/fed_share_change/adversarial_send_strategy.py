from functools import total_ordering
from pydoc import cli
import flwr as fl
import sys
import pandas as pd

sys.path.append("..")
from adversarial_strategy import AdversarialStrategy

from collections import OrderedDict

from flwr.common import (
    FitIns
)

SHARED_FILES = "../../../data/participants/change_shared/" #input path

class AdversarialStrategyChange(AdversarialStrategy):
    
    def configure_fit(
        self, rnd: int, parameters, client_manager
    ):
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)
        config["round"] = rnd
        print("Round:\t%d"%rnd)
        config["reload"] = SHARED_FILES+"round%d.csv"%rnd
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
