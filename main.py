import config
import game
from model import vamperouge_net
from self_play import SelfPlay

if __name__ == "__main__":
    neural_net = vamperouge_net(config)

    if config.load_model:
        neural_net.load_checkpoint(
            config.load_folder_file[0], config.load_folder_file[1]
        )

    self_play = SelfPlay(neural_net, config)
    if config.load_model:
        print("Load train samples from file")
        self_play.load_train_samples()
    self_play.learn()
