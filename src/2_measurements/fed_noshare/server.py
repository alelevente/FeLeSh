import flwr as fl

import sys
sys.path.append("..")

import adversarial_strategy

RESULT_PATH = "../../../results/fed_noshare/"

adv_strat = adversarial_strategy.AdversarialStrategy(n_client=10,
        result_path = RESULT_PATH,
        min_fit_clients=10,
        min_eval_clients=10,
        min_available_clients=10)

fl.server.start_server("[::]:9605",
        config={"num_rounds": 100},
        strategy = adv_strat)
