import math
import random
from dataclasses import dataclass

import numpy as np

import game


class MCTS:
    def __init__(self, neural_net, config):
        self.neural_net = neural_net
        self.config = config
        self.states_actions_Q = {}
        self.states_actions_N = {}
        self.states_N = {}
        self.states_P = {}

        self.states_ending_score = {}
        self.states_valid_moves = {}

    def get_move_probabilities(self, state, temp=1):
        """
        get the policy vector representing move probabilities from MCTS
        simulations from the given state
        """
        for _ in range(self.config.num_MCTS_sims):
            self.search(state)

        s = game.hash_state(state)
        counts = [
            self.states_actions_N.get((s, a), 0)
            for a in range(game.ACTION_SIZE)
        ]

        if temp == 0:
            best_move = np.argmax(counts)
            policy = [0] * len(counts)
            policy[best_move] = 1
            return policy

        counts = [x ** (1.0 / temp) for x in counts]
        counts_sum = float(sum(counts))
        return [x / counts_sum for x in counts]

    def search(self, state):
        """
        One iteration of MCTS. This method is recursively called until a
        leaf node is found.
        
        The action chosen at each node is the one with the maximum upper
        confidence bound.
        
        Returns:
            v: the negative of the value of the current state
        """
        s = game.hash_state(state)

        if s not in self.states_ending_score:
            self.states_ending_score[s] = game.get_state_score(state, 1)
        if self.states_ending_score[s] != 0:
            # terminal node: outcome propagated up the search path
            return -self.states_ending_score[s]

        # leaf node: neural net is used to get an initial policy and value for the state
        if s not in self.states_P:
            # transform state by using a randomly selected symmetry before it is evaluated
            # by the NN, so that the MC evaluation is averaged over different biases
            transformed_state = random.choice(game.get_symmetries(state))
            self.states_P[s], v = self.neural_net.predict(transformed_state)
            legal_moves = game.get_legal_moves(state, 1)
            # put 0 in the policy for illegal moves
            self.states_P[s] = self.states_P[s] * legal_moves
            # renormalize the policy
            policy_sum = self.states_P[s].sum().item()
            if policy_sum > 0:
                self.states_P[s] /= policy_sum
            else:
                # if all legal moves probabilities are 0, let all legal moves probabilities be equal
                # print something here as it is not expected to get this message often
                print(
                    "All legal moves probabilities are 0! Replacing with uniform distribution..."
                )
                self.states_P[s] = self.states_P[s] + legal_moves
                self.states_P[s] /= np.sum(self.states_P[s])

            self.states_valid_moves[s] = legal_moves
            self.states_N[s] = 0
            # the value is propagated up the search path
            return -v

        legal_moves = self.states_valid_moves[s]
        current_best = -float("inf")
        best_move = -1

        # pick the action with the highest upper confidence bound
        for a in range(game.ACTION_SIZE):
            if not legal_moves[a]:
                continue
            Q = self.states_actions_Q.get((s, a), 0)
            N = self.states_actions_N.get((s, a), 0)

            U = Q + self.config.cpuct * self.states_P[s][a] * math.sqrt(
                self.states_N[s]
            ) / (1 + N)

            if U > current_best:
                current_best = U
                best_move = a

        a = best_move
        next_state, next_player = game.get_next_state(state, 1, a)
        next_state = game.get_canonical_form(next_state, next_player)

        # the value is retrieved from the next state
        v = self.search(next_state)

        if (s, a) in self.states_actions_Q:
            self.states_actions_Q[(s, a)] = (
                self.states_actions_N[(s, a)] * self.states_actions_Q[(s, a)] + v
            ) / (self.states_actions_N[(s, a)] + 1)
            self.states_actions_N[(s, a)] += 1
        else:
            self.states_actions_Q[(s, a)] = v
            self.states_actions_N[(s, a)] = 1

        self.states_N[s] += 1
        # the value is propagated up the remaining of the search path
        return -v
