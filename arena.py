import time

import numpy as np

from progress.bar import Bar
from progress.misc import AverageMeter


class Arena:
    """
    Implementation of a battle between two agents.
    """

    def __init__(self, player1, player2, game):
        self.player1 = player1
        self.player2 = player2
        self.game = game

    def play_game(self):
        """
        Run one episode of a game

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        current_player = 1
        state = self.game.get_init_state()

        while self.game.get_state_score(state, current_player) == 0:
            move = players[current_player + 1](
                self.game.get_canonical_form(state, current_player)
            )
            legal_moves = self.game.get_legal_moves(
                self.game.get_canonical_form(state, current_player), 1
            )
            if legal_moves[move] == 0:
                print(move)
                assert legal_moves[move] > 0
            state, current_player = self.game.get_next_state(
                state, current_player, move
            )

        return current_player * self.game.get_state_score(state, current_player)

    def play_games(self, num):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            one_won: games won by player1
            two_won: games won by player2
            draws:  games won by nobody
        """
        episode_time = AverageMeter()
        bar = Bar("Arena.play_games", max=num)
        end = time.time()
        episode = 0
        max_episodes = int(num)

        num = int(num / 2)
        one_won = 0
        two_won = 0
        draws = 0
        for _ in range(num):
            game_result = self.play_game()
            if game_result == 1:
                one_won += 1
            elif game_result == -1:
                two_won += 1
            else:
                draws += 1

            episode += 1
            episode_time.update(time.time() - end)
            end = time.time()
            bar.suffix = "({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}".format(
                eps=episode,
                maxeps=max_episodes,
                et=episode_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
            )
            bar.next()

        self.player1, self.player2 = self.player2, self.player1

        for _ in range(num):
            game_result = self.play_game()
            if game_result == -1:
                one_won += 1
            elif game_result == 1:
                two_won += 1
            else:
                draws += 1

            eps += 1
            episode_time.update(time.time() - end)
            end = time.time()
            bar.suffix = "({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}".format(
                eps=episode,
                maxeps=max_episodes,
                et=episode_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
            )
            bar.next()

        bar.finish()

        return one_won, two_won, draws
