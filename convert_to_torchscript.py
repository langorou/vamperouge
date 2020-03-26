import argparse
import torch

import config
from model import vamperouge_net


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_filename")
    parser.add_argument("script_module_filename")
    args = parser.parse_args()

    # get instance of model
    model = vamperouge_net(config)
    model.load_checkpoint("models", args.model_filename)

    # get example input
    example = torch.rand(1, 3, config.board_width, config.board_height)

    # generate a torch.jit.ScriptModule via tracing
    traced_script_module = torch.jit.trace(model, example)

    # save it
    traced_script_module.save(args.script_module_filename)
