import client
import tools

import flwr as fl

import argparse

parser = argparse.ArgumentParser(
    description="Federated Learning - client runner tool")
parser.add_argument("name", help="name of client")
parser.add_argument("csv_data_file", help="Path to data set")
parser.add_argument("--keep_received", help="whether to keep received data",
                    action="store_true")

args = parser.parse_args()
name = args.name
path = args.csv_data_file

mnist_client = client.MnistClient(name, path, keep_received_data=args.keep_received)
fl.client.start_numpy_client("[::]:9605",client=mnist_client)

