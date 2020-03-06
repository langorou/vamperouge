import pytest
import warnings

from game import (
    State,
    Coordinates,
    Cell,
    Move,
    IllegalMoveException,
    HUMAN,
    VAMPIRE,
    WEREWOLF,
)


def test_apply_move():
    h = 3
    w = 4
    ######################
    # H:05#____#____#____#
    ######################
    # V:06#W:17#____#V:01#
    ######################
    # H:10#W:10#____#____#
    ######################
    grid = {
        Coordinates(0, 0): Cell(HUMAN, 5),
        Coordinates(1, 0): Cell(VAMPIRE, 6),
        Coordinates(2, 0): Cell(HUMAN, 10),
        Coordinates(1, 1): Cell(WEREWOLF, 17),
        Coordinates(2, 1): Cell(WEREWOLF, 10),
        Coordinates(1, 3): Cell(VAMPIRE, 1),
    }
    state = State(grid=grid, height=h, width=w)
    with pytest.raises(IllegalMoveException):
        state.apply_moves([Move(Coordinates(1, 0), 7, Coordinates(0, 0),)], VAMPIRE)
        assert state.get_cell(1, 0) == Cell(VAMPIRE, 6)
    with pytest.raises(IllegalMoveException):
        state.apply_moves([Move(Coordinates(1, 0), 6, Coordinates(0, 0),)], WEREWOLF)
        assert state.get_cell(1, 0) == Cell(VAMPIRE, 6)
    state.apply_moves([Move(Coordinates(1, 0), 6, Coordinates(0, 0),)], VAMPIRE)
    assert state.get_cell(1, 0) is None
    assert state.get_cell(0, 0) == Cell(VAMPIRE, 11)
    state.apply_moves([Move(Coordinates(1, 1), 17, Coordinates(0, 0),)], WEREWOLF)
    assert state.get_cell(1, 0) is None
    assert state.get_cell(0, 0) == Cell(WEREWOLF, 17)
    state.apply_moves([Move(Coordinates(1, 3), 1, Coordinates(2, 3),)], VAMPIRE)
    assert state.get_cell(1, 3) is None
    assert state.get_cell(2, 3) == Cell(VAMPIRE, 1)
    state.apply_moves([Move(Coordinates(2, 1), 10, Coordinates(2, 0),)], WEREWOLF)
    assert state.get_cell(2, 1) is None
    cell = state.get_cell(2, 0)
    if cell is not None:
        assert cell.race in (WEREWOLF, HUMAN)
        assert 1 <= cell.count <= 20
    else:
        warnings.warn("cell is None")
