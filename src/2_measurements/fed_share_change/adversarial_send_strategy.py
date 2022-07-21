from functools import total_ordering
from pydoc import cli
import flwr as fl
import sys
import pandas as pd

sys.path.append("..")
from adversarial_strategy import AdversarialStrategy

sys.path.append("../../1_sharing_methods")
from data_mixer import DataMixer


from collections import OrderedDict

from flwr.common import (
    FitIns
)

NON_IID50_PATH = "../../../data/participants/non_iid50/" # input
SAMPLES_TO_SHARE = 27 #num. samples/digit/clients

class AdversarialStrategyChange(AdversarialStrategy):
    
    def __init__(self, n_client, result_path, **kwargs):
        super().__init__(n_client, result_path, **kwargs)
        self.data_mixer = DataMixer(NON_IID50_PATH)
    
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
        config["reload"] = self.data_mixer.create_new_shared_data(SAMPLES_TO_SHARE).to_json()
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
