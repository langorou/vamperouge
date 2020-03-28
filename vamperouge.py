import argparse

import torch

import config
from tcp_client import TcpClient
from mcts import MCTS
from model import vamperouge_net

VERSION = "percival"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ip_addr")
    parser.add_argument("port")
    args = parser.parse_args()

    # create IA
    print(f"loading vamperouge_{VERSION}")
    model = vamperouge_net(config)
    model.load_checkpoint("models", f"{VERSION}.pth.tar")
    mcts = MCTS(model, config)
    # bind IA to client
    client = TcpClient(mcts)
    # start client
    client.connect(args.ip_addr, int(args.port))
    client.start()
