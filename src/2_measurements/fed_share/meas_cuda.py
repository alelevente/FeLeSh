import client
import torch

import flwr as fl

from concurrent.futures import ProcessPoolExecutor as Pool

INPUT_PATH = "../../../data/participants/non_iid50/"
N_PARTICIPANTS = 10
N_THREADS = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def client_runner(identifier):
    name = "client%d"%identifier
    path = INPUT_PATH+"participant%d.csv"%identifier
    mnist_client = client.MnistClientShare(name, path, device)
    fl.client.start_numpy_client("[::]:9605", client=mnist_client)

if __name__ == "__main__":
    with Pool(N_THREADS) as p:
        for i in p.map(client_runner, range(N_PARTICIPANTS)):
            print(i)
