from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass
from random import random, randint

import torch

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


class Game:
    MIN_ROW = 3
    MIN_COL = 3
    MAX_ROW = 16
    MAX_COL = 16
    MAX_MOVES = 200

    @staticmethod
    def get_action_size():
        return Game.MAX_COL * Game.MAX_ROW * 8

    @staticmethod
    def get_canonical_form(state, player):
        canonical_state = deepcopy(state)
        for cell in canonical_state.grid.values():
            cell.race *= player
        return canonical_state

    # TODO for now, generate only one map but should be random
    @staticmethod
    def get_init_state():
        # w = randint(Game.MIN_COL, Game.MAX_COL)
        # h = randint(Game.MIN_ROW, Game.MIN_COL)
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

    @staticmethod
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

    @staticmethod
    def get_legal_moves(state, player):
        return state.get_legal_moves_as_tensor(player)

    @staticmethod
    def get_next_state(state, player, action):
        state = state.apply_action(action, player)
        return state, -player

    @staticmethod
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
        # draw
        if state.n_moves >= Game.MAX_MOVES:
            return EPSILON
        player_units = 0
        opponent_units = 0
        for cell in state.grid.values():
            if cell.race == player:
                player_units += 1
            if cell.race == -player:
                opponent_units += 1
            # game not finished if units for both players and max moves not reached
            if (
                state.n_moves < Game.MAX_MOVES
                and player_units != 0
                and opponent_units != 0
            ):
                return 0
        # either one player has no unit left or the maximum number of moves has been reached
        if player_units > opponent_units:
            return 1
        if opponent_units > player_units:
            return -1
        # draw
        return EPSILON

    @staticmethod
    def get_symmetries(state, policy):
        # TODO generate symmetries
        return [(state, policy)]


class State:
    def __init__(self, grid=None, height=0, width=0, n_moves=0):
        # maps coordinates to a cell
        if grid is None:
            self.grid = {}
        else:
            self.grid = grid
        self.height = height
        self.width = width
        # let's say vampires go first
        self.current_player = VAMPIRE
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
        self.n_moves = 0

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

    def get_legal_moves_as_tensor(self, race):
        legal_moves = torch.zeros([Game.get_action_size()], dtype=torch.int32)
        for coord, cell in self.grid.items():
            if cell.race != race:
                continue
            for direction in self._generate_valid_directions(coord):
                legal_moves[8 * (coord.x * Game.MAX_ROW + coord.y) + direction] = 1
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
        x, y = divmod(encoded_coord, Game.MAX_ROW)
        start_coord = Coordinates(x, y)
        end_coord = self._transform_coordinates(start_coord, self.transforms[direction])
        move = Move(start_coord, self.grid[start_coord].count, end_coord)
        next_state = deepcopy(self)
        next_state.apply_moves([move], player)
        return next_state
