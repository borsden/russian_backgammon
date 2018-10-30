from typing import Dict

import pytest
import russian_backgammon as bg

#
# xChecker = bg.xChecker
# oChecker = bg.oChecker

Columns = Dict[int, int]


def get_board(x_columns: Columns = {}, y_columns: Columns = {}) -> bg.Board:
    """Method to create board from passed columns. """

    def fill_board(board: bg.Board, columns: Columns, checker_type: bg.Checker) -> None:
        for column, number in columns.items():
            board.cols[column] = [checker_type] * number

    board = bg.Board()
    board.cols = [[] for _ in range(board.NUM_COLS)]

    fill_board(board, x_columns, bg.xChecker)
    with board.reverse():
        fill_board(board, y_columns, bg.oChecker)

    # board.draw()
    assert board.check_correct(), 'This board is incorrect.'
    return board


@pytest.mark.parametrize(['columns', 'dice', 'expected_moves'], [
    # (
    #     (
    #         {0: 10, 4: 2, 5: 1, 6: 2},
    #         {0: 12, 13: 1, 14: 1, 15: 1},
    #     ),
    #     (6, 3),
    #     {
    #         ((0, 6), (5, 3)),
    #         ((0, 6), (4, 3)),
    #         ((0, 6), (6, 3)),
    #         ((4, 3), (4, 6)),
    #         ((4, 3), (5, 6)),
    #         ((4, 3), (7, 6)),
    #         ((4, 6), (5, 3)),
    #         ((4, 6), (6, 3)),
    #         ((4, 6), (10, 3)),
    #         ((4, 6), (4, 3)),
    #         ((5, 3), (8, 6)),
    #         ((5, 6), (11, 3)),
    #         ((5, 6), (6, 3)),
    #         ((6, 3), (9, 6))
    #
    #     }
    # ),
    # (
    #     (
    #         {0: 9, 6: 6},
    #         {0: 15},
    #     ),
    #     (6, 6),
    #     {
    #         ((0, 6),)
    #     }
    # ),
    # (
    #     (
    #         {0: 12, 10: 1, 11: 1, 13: 1},
    #         {0: 7, 2: 1, 3: 1, 5: 1, 6: 1, 7: 1, 8: 1, 15: 1, 16: 1},
    #     ),
    #     (4, 3),
    #     {
    #         ((10, 3),),
    #         ((13, 3),),
    #     }
    # ),
    # (
    #     (
    #         {0: 15},
    #         {0: 15},
    #     ),
    #     (5, 5),
    #     {
    #         ((0, 5), (0, 5)),
    #         ((0, 5), (5, 5), (10, 5), (15, 5))
    #     },
    # ),
    # (
    #     (
    #         {0: 14, 11: 1},
    #         {0: 11, 4: 1, 5: 1, 14: 1, 15: 1},
    #     ),
    #     (2, 3),
    #     {
    #         ((11, 2),),
    #         ((11, 3),)
    #     },
    # ),
    # (
    #     (
    #         {18: 2, 19: 2, 20: 2, 21: 2, 22: 1, 23: 1},
    #         {21: 1},
    #     ),
    #     (6, 5),
    #     {
    #         ((18, 5), (18, 6)),
    #         ((18, 6), (19, 5)),
    #         ((18, 6), (18, 5))
    #     },
    # ),
    # (
    #     (
    #         {0: 15},
    #         {0: 14, 3: 1},
    #     ),
    #     (3, 1),
    #     {
    #         ((0, 3), (3, 1)),
    #         ((0, 1), (1, 3)),
    #     },
    # ),
    (
        (
            {0: 9, 13: 1, 14: 1, 15: 2, 16: 1, 17: 1},
            {0: 13, 13: 1, 14: 1},
        ),
        (2, 1),
        {
            ((16, 1), (17, 2)),
            ((13, 1), (14, 2)),
            ((14, 2), (16, 1)),
            ((15, 1), (15, 2)),
            ((16, 2), (18, 1)),
            ((15, 2), (15, 1)),
            ((13, 2), (14, 1)),
            ((13, 2), (15, 1)),
            ((13, 1), (16, 2)),
            ((13, 1), (17, 2)),
            ((13, 2), (16, 1)),
            ((13, 2), (17, 1)),
            ((17, 1), (18, 2)),
            ((15, 2), (16, 1)),
            ((16, 2), (17, 1)),
            ((14, 2), (15, 1)),
            ((14, 2), (17, 1)),
            ((14, 1), (15, 2)),
            ((13, 1), (15, 2)),
            ((15, 1), (17, 2)),
            ((14, 1), (16, 2)),
            ((14, 1), (17, 2)),
            ((17, 2), (19, 1))}
        ,
    ),
])
def test_available_moves(columns, dice, expected_moves):
    board = get_board(*columns)

    game = bg.Game(players=[bg.Player(), bg.Player()])
    game.board = board
    available_moves = game.get_available_moves(dice, bg.xChecker)

    assert available_moves == set(expected_moves)
