import time
from typing import Set

import backgammon.game as bg
from backgammon.agents import RandomAgent

store = {
    'dice': [(3, 3), (6, 1), (3, 1), (3, 2), (5, 1), (1, 2), (3, 1), (6, 4), (3, 5), (6, 4), (3, 3), (2, 6), (6, 6),
             (5, 2), (3, 4), (2, 2), (6, 1), (3, 4), (3, 6), (4, 6), (1, 2), (3, 1), (4, 3), (2, 4), (1, 3), (3, 3),
             (5, 2), (3, 4), (4, 2), (3, 4), (6, 6), (2, 1), (3, 6), (1, 2), (1, 3), (2, 3), (6, 2), (6, 3), (4, 4),
             (6, 3), (5, 2), (5, 6), (2, 4), (6, 3), (3, 4), (6, 3), (3, 4), (4, 2), (3, 3), (6, 5), (5, 6), (3, 1),
             (6, 6), (2, 1), (1, 2), (6, 3), (5, 1), (1, 1), (1, 6), (5, 3), (4, 3), (5, 3), (1, 6), (5, 4), (6, 1),
             (2, 5), (5, 2), (1, 6), (5, 6), (6, 4), (5, 6), (5, 1), (6, 6), (5, 5), (6, 1), (3, 2), (1, 1), (5, 5),
             (4, 5), (3, 4), (1, 1), (6, 2), (5, 3), (4, 5), (3, 6), (3, 5), (4, 5), (2, 2), (6, 6), (3, 4), (5, 6),
             (4, 4), (5, 5)],
    'moves': [((0, 3), (3, 3), (6, 3)), ((0, 6), (6, 1)), ((0, 3), (3, 1)), ((0, 3), (7, 2)), ((4, 5), (9, 1)),
              ((0, 2), (2, 1)), ((0, 3), (9, 1)), ((0, 6), (9, 4)), ((3, 5), (8, 3)), ((6, 4), (10, 6)),
              ((0, 3), (3, 3), (10, 3), (10, 3)), ((0, 2), (2, 6)), ((0, 6), (11, 6), (13, 6), (17, 6)),
              ((8, 5), (13, 2)), ((6, 3), (9, 4)), ((0, 2), (2, 2), (15, 2), (17, 2)), ((13, 6), (19, 1)),
              ((13, 4), (17, 3)), ((0, 6), (6, 3)), ((3, 6), (9, 4)), ((19, 2), (20, 1)), ((3, 1), (13, 3)),
              ((6, 4), (10, 3)), ((16, 4), (20, 2)), ((0, 3), (13, 1)), ((4, 3), (7, 3), (10, 3), (16, 3)),
              ((0, 5), (9, 2)), ((13, 3), (16, 4)), ((0, 2), (2, 4)), ((0, 4), (19, 3)),
              ((0, 6), (3, 6), (11, 6), (13, 6)), ((4, 2), (22, 1)), ((0, 6), (17, 3)), ((0, 1), (20, 2)),
              ((0, 3), (21, 1)), ((1, 2), (3, 3)), ((0, 2), (14, 6)), ((0, 6), (20, 3)),
              ((5, 4), (9, 4), (9, 4), (19, 4)), ((0, 6), (19, 3)), ((2, 5), (20, 2)), ((0, 5),), ((0, 4), (7, 2)),
              ((4, 3), (7, 6)), ((0, 4), (3, 3)), ((6, 6), (12, 3)), ((4, 4), (4, 3)), ((5, 2), (13, 4)),
              ((6, 3), (6, 3), (13, 3), (13, 3)), ((0, 6), (7, 5)), ((16, 5), (16, 6)), ((0, 3), (6, 1)),
              ((7, 6), (8, 6), (14, 6)), ((0, 2), (22, 1)), ((20, 1), (20, 2)), ((2, 3), (7, 6)), ((9, 5), (13, 1)),
              ((0, 1), (12, 1), (13, 1), (14, 1)), ((14, 6), (21, 1)), ((15, 5), (20, 3)), ((9, 3), (12, 4)),
              ((3, 3), (15, 5)), ((6, 1), (9, 6)), ((1, 4), (17, 5)), ((6, 6), (22, 1)), ((5, 2), (13, 5)),
              ((7, 5), (20, 2)), ((6, 6), (22, 1)), ((15, 6), (16, 5)), ((6, 6), (18, 4)), ((14, 6),),
              ((7, 5), (20, 1)), ((12, 5), (12, 5), (12, 5), (17, 5)), ((12, 1), (13, 6)), ((17, 3), (20, 2)),
              ((21, 1), (21, 1), (21, 1), (22, 1)), ((17, 5),), ((12, 4), (16, 5)),
              ((20, 1), (21, 1), (23, 1), (23, 1)), ((6, 2), (6, 6)), ((19, 5), (21, 3)), ((8, 5), (13, 4)),
              ((21, 6), (22, 3)), ((5, 3), (8, 5)), ((22, 4), (22, 5)), ((12, 2), (13, 2), (17, 2), (19, 2)),
              ((22, 6), (22, 6), (22, 6), (22, 6)), ((14, 3), (15, 4)), ((22, 5), (23, 6)),
              ((17, 4), (19, 4), (21, 4), (21, 4)), ((23, 5),)],
    'who_start': 1
}

"""
11 10  9  8  7  6  5  4  3  2  1  0
-- -- -- -- -- -- -- -- -- -- -- --
 o  o  o                           
 o  o                              
 o  o                              
 o  o                              
 o  o                              
 o  o                              
 _  _  _  _  _  _  _  _  _  _  _  _
-- -- -- -- -- -- -- -- -- -- -- --
12 13 14 15 16 17 18 19 20 21 22 23

x
107.89, 99.2 - Mac / 49.9 / 8.3
29.92 - PK / 15.98 / 3
"""

all_dice = (dice for dice in store['dice'])
all_moves = (moves for moves in store['moves'])


class FakeRandomAgent(RandomAgent):
    def get_action(self, available_moves: Set[bg.Moves], board: bg.Board) -> bg.Moves:
        return next(all_moves)


def roll_dice(*args, **kwargs):
    return next(all_dice)


if __name__ == '__main__':
    t1 = time.time()
    players = (FakeRandomAgent(), FakeRandomAgent())
    game = bg.Game(players=players, show_logs=False, who_start=store['who_start'])
    bg.roll_dice = roll_dice

    status = game.play()

    print(game.board)
    print(status)
    print(game._store)
    print(time.time() - t1)
