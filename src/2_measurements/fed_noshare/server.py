import flwr as fl

import adversarial_strategy

adv_strat = adversarial_strategy.AdversarialStrategy(n_client=10,
        min_fit_clients=10,
        min_eval_clients=10,
        min_available_clients=10)

fl.server.start_server("[::]:9605",
        config={"num_rounds": 500},
        strategy = adv_strat)