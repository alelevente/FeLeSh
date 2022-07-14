import flwr as fl

import adversarial_strategy

adv_strat = adversarial_strategy.AdversarialStrategy(n_client=3,
        min_fit_clients=3,
        min_eval_clients=3,
        min_available_clients=3)

fl.server.start_server("[::]:9605",
        config={"num_rounds": 50},
        strategy = adv_strat)
