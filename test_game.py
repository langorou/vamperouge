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


def test_get_legal_moves():
    h = 3
    w = 4
    ######################
    # ____#V:07#____#____#
    ######################
    # ____#W:09#____#____#
    ######################
    # V:05#____#____#____#
    ######################
    grid = {
        Coordinates(0, 0): Cell(VAMPIRE, 5),
        Coordinates(1, 1): Cell(WEREWOLF, 9),
        Coordinates(1, 2): Cell(VAMPIRE, 7),
    }
    state = State(grid=grid, height=h, width=w)
    vampire_legal_moves = state.get_legal_moves(VAMPIRE)
    werewolf_legal_moves = state.get_legal_moves(WEREWOLF)
    assert len(vampire_legal_moves) == 8
    assert len(werewolf_legal_moves) == 8
    vampire_expected = [
        Move(Coordinates(0, 0), 5, Coordinates(1, 0)),
        Move(Coordinates(0, 0), 5, Coordinates(1, 1)),
        Move(Coordinates(0, 0), 5, Coordinates(0, 1)),
        Move(Coordinates(1, 2), 7, Coordinates(0, 2)),
        Move(Coordinates(1, 2), 7, Coordinates(0, 1)),
        Move(Coordinates(1, 2), 7, Coordinates(1, 1)),
        Move(Coordinates(1, 2), 7, Coordinates(2, 1)),
        Move(Coordinates(1, 2), 7, Coordinates(2, 2)),
    ]
    werewolf_expected = [
        Move(Coordinates(1, 1), 9, Coordinates(2, 1)),
        Move(Coordinates(1, 1), 9, Coordinates(2, 2)),
        Move(Coordinates(1, 1), 9, Coordinates(1, 2)),
        Move(Coordinates(1, 1), 9, Coordinates(0, 2)),
        Move(Coordinates(1, 1), 9, Coordinates(0, 1)),
        Move(Coordinates(1, 1), 9, Coordinates(0, 0)),
        Move(Coordinates(1, 1), 9, Coordinates(1, 0)),
        Move(Coordinates(1, 1), 9, Coordinates(2, 0)),
    ]
    for m in vampire_expected:
        assert m in vampire_legal_moves
    for m in werewolf_expected:
        assert m in werewolf_legal_moves


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
        Coordinates(0, 2): Cell(HUMAN, 5),
        Coordinates(0, 1): Cell(VAMPIRE, 6),
        Coordinates(0, 0): Cell(HUMAN, 10),
        Coordinates(1, 1): Cell(WEREWOLF, 17),
        Coordinates(1, 0): Cell(WEREWOLF, 10),
        Coordinates(3, 1): Cell(VAMPIRE, 1),
    }
    state = State(grid=grid, height=h, width=w)
    with pytest.raises(IllegalMoveException):
        state.apply_moves([Move(Coordinates(0, 1), 7, Coordinates(0, 2),)], VAMPIRE)
        assert state.get_cell(0, 1) == Cell(VAMPIRE, 6)
    with pytest.raises(IllegalMoveException):
        state.apply_moves([Move(Coordinates(0, 1), 6, Coordinates(0, 2),)], WEREWOLF)
        assert state.get_cell(0, 1) == Cell(VAMPIRE, 6)
    state.apply_moves([Move(Coordinates(0, 1), 6, Coordinates(0, 2),)], VAMPIRE)
    assert state.get_cell(0, 1) is None
    assert state.get_cell(0, 2) == Cell(VAMPIRE, 11)
    state.apply_moves([Move(Coordinates(1, 1), 17, Coordinates(0, 2),)], WEREWOLF)
    assert state.get_cell(0, 1) is None
    assert state.get_cell(0, 2) == Cell(WEREWOLF, 17)
    state.apply_moves([Move(Coordinates(3, 1), 1, Coordinates(3, 0),)], VAMPIRE)
    assert state.get_cell(3, 1) is None
    assert state.get_cell(3, 0) == Cell(VAMPIRE, 1)
    state.apply_moves([Move(Coordinates(1, 0), 10, Coordinates(0, 0),)], WEREWOLF)
    assert state.get_cell(1, 0) is None
    cell = state.get_cell(0, 0)
    if cell is not None:
        assert cell.race in (WEREWOLF, HUMAN)
        assert 1 <= cell.count <= 20
    else:
        warnings.warn("cell is None")
