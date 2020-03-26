import os
import sys
import time
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np

import game
from arena import Arena
from mcts import MCTS
from model import vamperouge_net
from progress.bar import Bar
from progress.misc import AverageMeter


class SelfPlay:
    """
    Implementation of the self-play and training of the neural network.
    """

    def __init__(self, neural_net, config):
        self.neural_net = neural_net
        # competitor neural network
        self.competitor_nn = vamperouge_net(config)
        self.config = config
        self.mcts = MCTS(neural_net, config)
        self.train_samples_history = []
        self.skip_first_self_play = False

    def run_episode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        train_samples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in train_samples.
        It uses a temp=1 if episode_step < tempThreshold, and thereafter
        uses temp=0.
        Returns:
            train_samples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        train_samples = []
        state = game.get_init_state()
        current_player = 1
        episode_step = 0

        while True:
            episode_step += 1
            canon_state = game.get_canonical_form(state, current_player)
            temp = int(episode_step < self.config.temperature_threshold)

            policy = self.mcts.get_move_probabilities(canon_state, temp=temp)
            sym = game.get_symmetries(canon_state, policy)
            for s, p in sym:
                train_samples.append([s, current_player, p, None])

            move = np.random.choice(len(policy), p=policy)
            state, current_player = game.get_next_state(
                state, current_player, move
            )

            r = game.get_state_score(state, current_player)

            if r != 0:
                return [
                    (s, pcy, r * ((-1) ** (pyr != current_player)))
                    for s, pyr, pcy, _ in train_samples
                ]

    def learn(self):
        """
        Performs num_iters iterations with num_eps episodes of self-play in each
        iteration. After every iteration, retrains neural network with
        examples in train_samples (which has a maximum length of max_queue_length).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= update_threshold fraction of games.
        """

        for i in range(1, self.config.num_iters + 1):
            print("------iteration " + str(i) + "------")
            if not self.skip_first_self_play or i > 1:
                iteration_train_samples = deque([], maxlen=self.config.max_queue_length)

                episode_time = AverageMeter()
                bar = Bar("Self Play", max=self.config.num_eps)
                end = time.time()

                for episode in range(self.config.num_eps):
                    # reset search tree
                    self.mcts = MCTS(self.neural_net, self.config)
                    iteration_train_samples += self.run_episode()

                    episode_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix = "({ep}/{max_ep}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}".format(
                        ep=episode + 1,
                        max_ep=self.config.num_eps,
                        et=episode_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                    )
                    bar.next()
                bar.finish()

                # save the iteration examples to the history
                self.train_samples_history.append(iteration_train_samples)

            if (
                len(self.train_samples_history)
                > self.config.num_iters_for_train_samples_history
            ):
                print(
                    "len(train_samples_history) =",
                    len(self.train_samples_history),
                    " => remove the oldest train_samples",
                )
                self.train_samples_history.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.save_train_samples(i - 1)

            # shuffle examples before training
            train_samples = []
            for e in self.train_samples_history:
                train_samples.extend(e)
            shuffle(train_samples)

            # training new network, keeping a copy of the old one
            self.neural_net.save_checkpoint(
                folder=self.config.checkpoint, filename="temp.pth.tar"
            )
            self.competitor_nn.load_checkpoint(
                folder=self.config.checkpoint, filename="temp.pth.tar"
            )
            previous_mcts = MCTS(self.competitor_nn, self.config)

            self.neural_net.train_from_samples(train_samples)
            new_mcts = MCTS(self.neural_net, self.config)

            print("battling against previous version")
            arena = Arena(
                lambda x: np.argmax(previous_mcts.get_move_probabilities(x, temp=0)),
                lambda x: np.argmax(new_mcts.get_move_probabilities(x, temp=0)),
            )
            prev_wins, new_wins, draws = arena.play_games(self.config.arena_compare)

            print("new/prev wins : %d / %d ; draws : %d" % (new_wins, prev_wins, draws))
            if (
                prev_wins + new_wins == 0
                or float(new_wins) / (prev_wins + new_wins)
                < self.config.update_threshold
            ):
                print("rejecting new model")
                self.neural_net.load_checkpoint(
                    folder=self.config.checkpoint, filename="temp.pth.tar"
                )
            else:
                print("accepting new model")
                self.neural_net.save_checkpoint(
                    folder=self.config.checkpoint, filename=self.get_checkpoint_file(i)
                )
                self.neural_net.save_checkpoint(
                    folder=self.config.checkpoint, filename="best.pth.tar"
                )

    def get_checkpoint_file(self, iteration):
        return "checkpoint_" + str(iteration) + ".pth.tar"

    def save_train_samples(self, iteration):
        folder = self.config.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(
            folder, self.get_checkpoint_file(iteration) + ".samples"
        )
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.train_samples_history)
        f.closed

    def load_train_samples(self):
        model_file = os.path.join(
            self.config.load_folder_file[0], self.config.load_folder_file[1]
        )
        samples_file = model_file + ".samples"
        if not os.path.isfile(samples_file):
            print(samples_file)
            r = input("File with train samples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with train samples found. Read it.")
            with open(samples_file, "rb") as f:
                self.train_samples_history = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skip_first_self_play = True
