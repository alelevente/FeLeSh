import flwr as fl

import adversarial_send_strategy

RESULT_PATH = "../../../results/fed_fixshare/"

adv_strat = adversarial_send_strategy.AdversarialStrategyFix(n_client=8,
        result_path = RESULT_PATH,
        min_fit_clients=8,
        min_eval_clients=8,
        min_available_clients=8)

fl.server.start_server("[::]:9605",
        config={"num_rounds": 100},
        strategy = adv_strat)
