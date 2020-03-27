from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass
from random import randint, random

import numpy as np
import torch
from numba import jit

HUMAN = 0
VAMPIRE = 1
WEREWOLF = -1

EPSILON = 1e-6


@dataclass(eq=True, frozen=True)
class Coordinates:
    x: int
    y: int


@dataclass
class Cell:
    race: int
    count: int


@dataclass
class Move:
    start: Coordinates
    n: int
    end: Coordinates


class IllegalMoveException(Exception):
    pass


MIN_ROW = 3
MIN_COL = 3
MAX_ROW = 16
MAX_COL = 16
MAX_MOVES = 120
ACTION_SIZE = MAX_COL * MAX_ROW * 8


def get_canonical_form(state, player):
    canonical_state = State(
        grid=deepcopy(state.grid),
        height=state.height,
        width=state.width,
        n_moves=state.n_moves,
    )
    for cell in canonical_state.grid.values():
        cell.race *= player
    return canonical_state


# TODO for now, generate only one map but should be random
def get_init_state():
    # w = randint(MIN_COL, MAX_COL)
    # h = randint(MIN_ROW, MIN_COL)
    h = 5
    w = 10
    grid = {
        Coordinates(2, 2): Cell(HUMAN, 4),
        Coordinates(9, 0): Cell(HUMAN, 2),
        Coordinates(9, 2): Cell(HUMAN, 1),
        Coordinates(9, 4): Cell(HUMAN, 2),
        Coordinates(4, 1): Cell(WEREWOLF, 4),
        Coordinates(4, 3): Cell(VAMPIRE, 4),
    }
    return State(grid=grid, height=h, width=w)


def hash_state(state):
    h = str(state.height) + str(state.width)
    h += str(state.n_moves)
    for x in range(state.width):
        for y in range(state.height):
            cell = state.get_cell(x, y)
            if cell is None:
                continue
            h += str(x) + str(y) + str(cell.race) + str(cell.count)
    return h


def get_legal_moves(state, player):
    return state.get_legal_moves_as_ndarray(player)


def get_next_state(state, player, action):
    state = state.apply_action(action, player)
    return state, -player


def get_state_score(state, player):
    """
    Input:
        state: current state
        player: current player (1 or -1)
    Returns:
        r: 0 if game has not ended
            1 if player won
            -1 if player lost
    """
    player_units = 0
    opponent_units = 0
    for cell in state.grid.values():
        if cell.race == player:
            player_units += 1
        if cell.race == -player:
            opponent_units += 1
        # game not finished if units for both players and max moves not reached
        if state.n_moves < MAX_MOVES and player_units != 0 and opponent_units != 0:
            return 0
    # either one player has no unit left or the maximum number of moves has been reached
    if player_units > opponent_units:
        return 1
    if opponent_units > player_units:
        return -1
    # draw
    return EPSILON


def _grid_to_ndarray(state):
    board = np.ndarray([state.width, state.height], dtype=Cell)
    for coord, cell in state.grid.items():
        board[coord.x, coord.y] = Cell(cell.race, cell.count)
    return board


@jit
def _reencode_policy(policy, width, height):
    """
    reencode policy as if width and height were the maximum dimensions
    """
    new_policy = width * height * 8 * [0]
    for i, prob in enumerate(policy):
        encoded_coord, direction = divmod(i, 8)
        x, y = divmod(encoded_coord, MAX_ROW)
        if x >= width or y >= height:
            continue
        new_policy[8 * (x * height + y) + direction] = prob
    return new_policy


@jit
def _encodeback_policy(policy, width, height):
    """
    encode back a reencoded policy
    """
    new_policy = MAX_COL * MAX_ROW * 8 * [0]
    for x in range(MAX_COL):
        for y in range(MAX_ROW):
            for d in range(8):
                if x >= width or y >= height:
                    new_policy[8 * (x * MAX_ROW + y) + d] = 0
                    continue
                new_policy[8 * (x * MAX_ROW + y) + d] = policy[8 * (x * height + y) + d]
    return new_policy


@jit
def _policy_to_ndarray(policy, width, height):
    new_policy = _reencode_policy(policy, width, height)
    policy_board = np.reshape(new_policy, (width, height, 8))
    policy_4d = np.ndarray([width, height, 3, 3], dtype=np.int32)
    for x, col in enumerate(policy_board):
        for y, probs in enumerate(col):
            policy_4d[x, y, 0, 0] = probs[0]
            policy_4d[x, y, 1, 0] = probs[1]
            policy_4d[x, y, 2, 0] = probs[2]
            policy_4d[x, y, 0, 1] = probs[7]
            policy_4d[x, y, 1, 1] = 0
            policy_4d[x, y, 2, 1] = probs[3]
            policy_4d[x, y, 0, 2] = probs[6]
            policy_4d[x, y, 1, 2] = probs[5]
            policy_4d[x, y, 2, 2] = probs[4]
    return policy_4d


@jit
def _ndarray_to_policy(policy_4d):
    policy = policy_4d.shape[0] * policy_4d.shape[1] * 8 * [0]
    for x, col in enumerate(policy_4d):
        for y, tile in enumerate(col):
            policy[8 * (x * policy_4d.shape[1] + y) + 0] = tile[0, 0]
            policy[8 * (x * policy_4d.shape[1] + y) + 1] = tile[1, 0]
            policy[8 * (x * policy_4d.shape[1] + y) + 2] = tile[2, 0]
            policy[8 * (x * policy_4d.shape[1] + y) + 7] = tile[0, 1]
            policy[8 * (x * policy_4d.shape[1] + y) + 3] = tile[2, 1]
            policy[8 * (x * policy_4d.shape[1] + y) + 6] = tile[0, 2]
            policy[8 * (x * policy_4d.shape[1] + y) + 5] = tile[1, 2]
            policy[8 * (x * policy_4d.shape[1] + y) + 4] = tile[2, 2]
    return _encodeback_policy(policy, policy_4d.shape[0], policy_4d.shape[1])


@jit
def _ndarray_to_grid(board):
    grid = {}
    for x, col in enumerate(board):
        for y, cell in enumerate(col):
            if cell is None:
                continue
            grid[Coordinates(x, y)] = Cell(cell.race, cell.count)
    return grid


def get_symmetries(state, policy=None):
    board = _grid_to_ndarray(state)
    if policy is not None:
        policy_board = _policy_to_ndarray(policy, state.width, state.height)
    symmetries = []

    for rot in range(1, 5):
        for mirrored in (False, True):
            sym_board = np.rot90(board, rot)
            if policy is not None:
                sym_policy_board = np.rot90(policy_board, rot)
                sym_policy_board = np.reshape(
                    [np.rot90(tile, rot) for col in sym_policy_board for tile in col],
                    (sym_board.shape[0], sym_board.shape[1], 3, 3),
                )
            if mirrored:
                sym_board = np.fliplr(sym_board)
                if policy is not None:
                    sym_policy_board = np.fliplr(sym_policy_board)
                    sym_policy_board = np.reshape(
                        [np.fliplr(tile) for col in sym_policy_board for tile in col],
                        (sym_board.shape[0], sym_board.shape[1], 3, 3),
                    )
            sym_state = State(
                grid=_ndarray_to_grid(sym_board),
                height=sym_board.shape[1],
                width=sym_board.shape[0],
                n_moves=state.n_moves,
            )
            if policy is not None:
                sym_policy = _ndarray_to_policy(sym_policy_board)
                symmetries.append((sym_state, sym_policy))
            else:
                symmetries.append(sym_state)

    return symmetries


class State:
    def __init__(self, grid=None, height=0, width=0, n_moves=0):
        # maps coordinates to a cell
        if grid is None:
            self.grid = {}
        else:
            self.grid = grid
        self.height = height
        self.width = width
        # directions for moves
        self.transforms = [
            (-1, -1),
            (0, -1),
            (1, -1),
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
        ]
        # number of moves played
        self.n_moves = n_moves

    def get_cell(self, x, y):
        return self.grid.get(Coordinates(x, y))

    def _transform_coordinates(self, coord, transform):
        t_x, t_y = transform
        x_res = coord.x + t_x
        if (x_res < 0) or (x_res >= self.width):
            return None
        y_res = coord.y + t_y
        if (y_res < 0) or (y_res >= self.height):
            return None
        return Coordinates(x=x_res, y=y_res)

    def _generate_moves_from_cell(self, coord):
        moves = []
        for t in self.transforms:
            target = self._transform_coordinates(coord, t)
            if target is None:
                continue
            moves.append(Move(coord, self.grid[coord].count, target))
        return moves

    def get_legal_moves(self, race):
        moves = []
        for coord, cell in self.grid.items():
            if cell.race != race:
                continue
            moves.extend(self._generate_moves_from_cell(coord))
        return moves

    def _is_coord_transform_valid(self, coord, transform):
        t_x, t_y = transform
        x_res = coord.x + t_x
        if (x_res < 0) or (x_res >= self.width):
            return False
        y_res = coord.y + t_y
        if (y_res < 0) or (y_res >= self.height):
            return False
        return True

    def _generate_valid_directions(self, coord):
        moves = []
        for i, t in enumerate(self.transforms):
            if not self._is_coord_transform_valid(coord, t):
                continue
            moves.append(i)
        return moves

    def get_legal_moves_as_ndarray(self, race):
        legal_moves = np.zeros([ACTION_SIZE], dtype=np.int32)
        for coord, cell in self.grid.items():
            if cell.race != race:
                continue
            for direction in self._generate_valid_directions(coord):
                legal_moves[8 * (coord.x * MAX_ROW + coord.y) + direction] = 1
        return legal_moves

    def _apply_move(self, race, target, count):
        end_cell = self.grid.get(target, Cell(race=race, count=0))
        if end_cell.count == 0 or end_cell.race == race:
            # no battle
            end_cell.count += count
            end_cell.race = race
            self.grid[target] = end_cell
            return
        # fight
        is_neutral = end_cell.race == HUMAN
        p = self._win_probability(count, end_cell.count, is_neutral)
        outcome = random()
        # automatic win
        if p == 1:
            # convert all humans
            end_cell.count = count + (end_cell.count if is_neutral else 0)
            end_cell.race = race
            self.grid[target] = end_cell
            return
        # win
        if outcome <= p:
            # each ally has probability p to survive. Against neutral, we have a probability p to convert them
            new_count = 0
            for _ in range(count):
                new_count += int(random() <= p)
            if is_neutral:
                for _ in range(end_cell.count):
                    new_count += int(random() <= p)
            end_cell.count = new_count
            end_cell.race = race
            if new_count:
                self.grid[target] = end_cell
            else:
                del self.grid[target]
            return
        # loss
        new_count = 0
        for _ in range(end_cell.count):
            new_count += int(random() <= 1 - p)
            end_cell.count = new_count
        if new_count:
            self.grid[target] = end_cell
        else:
            del self.grid[target]

    def _win_probability(self, e1, e2, is_e2_neutral):
        if e1 == e2:
            return 0.5
        if (is_e2_neutral and e1 > e2) or ((not is_e2_neutral) and (e1 > 1.5 * e2)):
            return 1
        if e1 < e2:
            return e1 / (2 * e2)
        return e1 / e2 - 0.5

    def apply_moves(self, moves, race):
        # sort moves by target cell
        moves.sort(key=lambda move: (move.end.x, move.end.y))
        last_end_coordinates = moves[0].end
        count = 0
        for move in moves:
            # ensure that the move is legal
            if self.grid[move.start].race != race:
                raise IllegalMoveException(
                    f"race {race} can't move {self.grid[move.start].race}"
                )
            # Aggregate by destination cell to take into account the case where 2 groups of race A
            # are going on cell of race B. We don't want two battles to occur, but only one, and
            # the # of A should be the sum of the 2 groups.
            cell = self.grid[move.start]
            if cell.count == move.n:
                del self.grid[move.start]
            elif cell.count >= move.n:
                cell.count -= move.n
                self.grid[move.start] = cell
            else:
                raise IllegalMoveException(f"move: {move}")

            # if target cell is the same: continue aggregating
            if move.end == last_end_coordinates:
                count += move.n
                continue

            # target cell is no more the same: compute possible states
            self._apply_move(race, last_end_coordinates, count)
            count = 0
        # apply the remaining moves
        self._apply_move(race, last_end_coordinates, count)
        self.n_moves += 1

    def apply_action(self, action, player):
        encoded_coord, direction = divmod(action, 8)
        x, y = divmod(encoded_coord, MAX_ROW)
        start_coord = Coordinates(x, y)
        end_coord = self._transform_coordinates(start_coord, self.transforms[direction])
        move = Move(start_coord, self.grid[start_coord].count, end_coord)
        next_state = State(
            grid=deepcopy(self.grid),
            height=self.height,
            width=self.width,
            n_moves=self.n_moves,
        )
        next_state.apply_moves([move], player)
        return next_state
