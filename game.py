from collections import namedtuple
from dataclasses import dataclass
from random import random

HUMAN = 0
VAMPIRE = 1
WEREWOLF = 2


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


class State:
    def __init__(self, grid=None, height=0, width=0):
        # maps coordinates to a cell
        if grid is None:
            self.grid = {}
        else:
            self.grid = grid
        self.height = height
        self.width = width

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
        transforms = [
            (0, -1),
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
        ]
        for t in transforms:
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
