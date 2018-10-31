import abc
import copy
import itertools
import random
from contextlib import contextmanager
from typing import List, Tuple, NamedTuple, Sequence, Set, Iterator, Any


class MoveError(Exception):
    pass


class Player:
    """Player."""
    checker_type: 'Checker'
    """Checker type for this player."""

    @abc.abstractmethod
    def get_action(self, available_moves: Set['Moves'], game: 'Game') -> 'Moves':
        """Strategy of player."""


class RandomPlayer(Player):
    """Random Player."""

    def get_action(self, available_moves: Set['Moves'], game: 'Game') -> 'Moves':
        return random.choice(list(available_moves))


class Checker(NamedTuple):
    """Checker."""
    type: str

    def __str__(self):
        return self.type


Dice = Tuple[int, int]
"""Roll of the dice."""

Column = List[Checker]
"""Column of a board. May include some checkers."""

Move = Tuple[int, int]
"""Move on a board. Include start position and die."""

Moves = Tuple[Move]
"""Moves in one step."""

xChecker = Checker(type=' x')
oChecker = Checker(type=' o')


class Board:
    """Backgammon board."""
    NUM_COLS = 24
    """Number of columns."""
    NUM_CHECKERS = 15
    """Number of checkers for one player."""
    CHECKER_TYPES = (xChecker, oChecker)

    def __init__(self) -> None:
        self.cols = [[] for _ in range(self.NUM_COLS)]
        """Columns of a board."""

        self.cols[0] = [self.CHECKER_TYPES[0] for _ in range(self.NUM_CHECKERS)]
        self.cols[self.half_cols_len] = [self.CHECKER_TYPES[1] for _ in range(self.NUM_CHECKERS)]
        """Add checkers at the heads."""

    def get_occupied_positions(self, checker_type: Checker) -> Iterator[int]:
        """Get positions occupied by this checker type.

        :param checker_type: checker type
        """
        return (index for index, col in enumerate(self.cols) if col and col[0] == checker_type)

    def check_position_available(self, checker_type: Checker, position: int) -> bool:
        """Check, that this position is available for this checker type.
        Also available positions out of board. It means checker withdrawal.

        :param checker_type: checker type
        :param position: position
        """
        if position < len(self.cols):
            col = self.cols[position]
            return not col or col[0] == checker_type
        else:
            return True

    @property
    def half_cols_len(self):
        """Half of total cols number."""
        return len(self.cols) // 2

    def copy(self) -> 'Board':
        """Create new instance of board."""
        b = Board()
        b.cols = copy.deepcopy(self.cols)
        return b

    def draw(self) -> None:
        """Draw current board."""

        def half_row() -> List[List[str]]:
            """Make quadrants of board with current position."""
            max_length = max((len(col) for col in self.cols[:self.half_cols_len]))
            rows = [
                [
                    str(column[checker_pos]) if len(column) > checker_pos else '  '
                    for column in self.cols[:self.half_cols_len]
                ]
                for checker_pos in range(max_length)
            ]
            return rows

        def col_pos(second_half: bool = False) -> List[str]:
            """Column position. """
            return [str(n + (self.half_cols_len if second_half else 0)).rjust(2, ' ') for n in
                    range(self.half_cols_len)]

        def horizontal_delimiter():
            """Simple separate positions and checkers."""
            return ['--' for _ in range(self.half_cols_len)]

        rows = [
            col_pos(),
            horizontal_delimiter(),
            *half_row(),
            [' _' for _ in range(self.half_cols_len)],
        ]

        # reverse second part of board.
        with self.reverse():
            reversed_rows = [
                *reversed(half_row()),
                horizontal_delimiter(),
                col_pos(True),
            ]

            rows = [
                *[list(reversed(row)) for row in rows],
                *reversed_rows
            ]
            for r in rows:
                print(' '.join(r))

    @contextmanager
    def viewpoint(self, checker_type: Checker) -> 'Board':
        """Context manager for turning a board, so head of passed checker_type will be at 0 position.

        :param checker_type: if True, that we don't what really reverse board.
        """
        if checker_type == self.CHECKER_TYPES[0]:
            yield self
        else:
            with self.reverse():
                yield self

    @contextmanager
    def reverse(self) -> 'Board':
        """Context manager to turn board and return after."""
        half_len = self.half_cols_len
        self.cols[half_len:], self.cols[:half_len] = self.cols[:half_len], self.cols[half_len:]
        yield self
        self.cols[half_len:], self.cols[:half_len] = self.cols[:half_len], self.cols[half_len:]

    def move(self, *moves: Move) -> None:
        """Move checkers.

        May multiple moves.
        """
        def _move(move: Move) -> Checker:
            """Move one checker from position with step."""
            from_pos, die = move
            to_pos = from_pos + die

            if not self.cols[from_pos]:
                raise MoveError('`From` position is empty.')

            _checker = self.cols[from_pos][0]

            if to_pos < len(self.cols):
                new_col = self.cols[to_pos]
                if new_col and new_col[0] != _checker:
                    raise MoveError('`To` position is already opponent position.')
                new_col.append(_checker)
            else:
                if not self.can_withdraw(_checker):
                    raise MoveError('Tried to withdraw checker when not all checkers in the home.')
                if to_pos != len(self.cols):
                    prev_positions = [pos for pos in self.get_occupied_positions(_checker) if pos < from_pos]
                    if prev_positions:
                        raise MoveError('Previous checkers can be withdraw.')

            del self.cols[from_pos][0]

            return _checker

        def _rollback(*moves: Move):
            """Rollback all moves. Simple use `to` position and negative step."""
            rollback_moves = ((move[0] + move[1], -move[1]) for move in reversed(moves))
            for move in rollback_moves:
                _move(move)

        if not moves:
            raise MoveError('Must be at least one move in args.')

        for move in moves:
            checker_type = _move(move)

        if not self.can_make_blocks(checker_type) and self.has_block(checker_type):
            _rollback(*moves)
            raise MoveError('Block is unavailable while opponent has not at least one checker in home.')

    @contextmanager
    def temp_move(self, *moves: Move) -> 'Board':
        """Temp move checkers contextmanager. After we return checkers, where they were."""
        cols = copy.deepcopy(self.cols)
        try:
            self.move(*moves)
            yield self
        except Exception as e:
            raise e
        finally:
            self.cols = cols

    def has_block(self, checker_type: Checker, block_size: int = 6) -> bool:
        """Check if there are any blocks (6 checkers in row). """

        def possible_blocks(iterable: Iterator[Any]) -> Iterator[Any]:
            """s -> (s0,s1, s2, s3, s4, s5), (s1,s2, s3, s4, s5, s6), ..."""
            _iterators = itertools.tee(iterable, block_size)
            for index, _iter in enumerate(_iterators):
                for i in range(index):
                    next(_iter, None)
            return zip(*_iterators)

        return any(
            (block[-1] - block[0]) == (block_size - 1)
            for block in possible_blocks(self.get_occupied_positions(checker_type))
        )

    def can_make_blocks(self, checker_type: Checker) -> bool:
        """Check that blocks is available.

        If opponent has at least one checker in his home, than available.
        """
        opponent_checker_type = [
            _checker_type
            for _checker_type in self.CHECKER_TYPES
            if _checker_type != checker_type
        ][0]
        opponent_home = (self.NUM_COLS // 4, self.NUM_COLS // 2)

        opponents_positions = self.get_occupied_positions(opponent_checker_type)

        return any(opponent_home[0] <= pos < opponent_home[1] for pos in opponents_positions)

    def check_move_available(self, *moves: Move) -> bool:
        """Check, that moves are available."""
        try:
            # We can get only one checker from the head.
            if len([move for move in moves if move[0] == 0]) > 1:
                return False

            with self.temp_move(*moves):
                return True
        except MoveError:
            return False

    def can_withdraw(self, checker_type: Checker) -> bool:
        """Check, that all checkers are in home."""
        home_pos = 3 * self.NUM_COLS // 4
        return not any(pos < home_pos for pos in self.get_occupied_positions(checker_type) )

    def check_correct(self):
        """Simple check that board is correct.
         It means, that we have 15 checkers for each type of checkers (except situations, when all checkers are in home).
        """
        for checker_type in self.CHECKER_TYPES:
            with self.viewpoint(checker_type):
                if not self.can_withdraw(checker_type):
                    positions = self.get_occupied_positions(checker_type)
                    total_checkers = sum(len(self.cols[pos]) for pos in positions)
                    if total_checkers != self.NUM_CHECKERS:
                        return False
        return True


class Game:
    """Game. """

    def __init__(self, players: List[Player], logs: bool = False) -> None:
        self.logs = logs

        self.board = Board()
        """Board."""
        self.players = players
        """Players of current game."""

        for player, checker_type in zip(self.players, self.board.CHECKER_TYPES):
            player.checker_type = checker_type

    def play(self) -> Player:
        """Play game. Return winner after end."""
        players_steps = itertools.cycle(self.players if random.randint(0, 1) else reversed(self.players))
        """Random select, which player start game. """

        while True:
            current_player = next(players_steps)
            self.make_step(player=current_player)
            if self.was_finished(current_player):
                break
        return current_player

    @property
    def is_end(self):
        """Check, that game was end."""
        # raise NotImplementedError
        return False

    def print(self, *values: Any) -> None:
        """Print value if logs flag is enabled."""
        if self.logs:
            print(values)

    def make_step(self, player: Player) -> None:
        """Roll dice, get available moves for them and get action from player.

        :param player: current Player.
        """
        dice = self.roll_dice()

        if self.logs:
            self.board.draw()

        with self.board.viewpoint(player.checker_type):

            available_moves = self.get_available_moves(dice, player.checker_type)

            self.print('Dice:', dice)
            self.print('Available moves:', available_moves)

            if available_moves:
                moves = player.get_action(available_moves, self)
                self.board.move(*moves)

                self.print('Move:', moves)

    def was_finished(self, player: Player) -> bool:
        """If there are no checkers on board, then player was finished."""
        return not list(self.board.get_occupied_positions(player.checker_type))

    def roll_dice(self) -> Dice:
        """Roll dice and return their values."""
        return random.randint(1, 6), random.randint(1, 6)

    def get_available_moves(self, dice: Dice, checkers_type: Checker) -> Set[Moves]:
        """Return available moves for current player with that dices.

        :param dice: dice
        :param checkers_type: current Player checkers type
        """

        def _flatten(seq):
            """Flat sequence, if it has subsequiences."""
            for el in seq:
                if any(isinstance(sub_el, Sequence) for sub_el in el):
                    yield from _flatten(el)
                else:
                    yield el

        def _get_moves(_dice: Sequence[int], from_positions: Set[int]) -> Set[Moves]:
            """Find all available moves.
            Iterate for every available position and every die in dice.
            If step with this die from this position is available, then we add this position to existing positions
            and recursive check other positions and dice.


            :param _dice: checking dice
            :param from_positions: available positions for steps
            :return:
            """
            #
            for pos in from_positions:
                for die in _dice:
                    new_pos = pos + die

                    if self.board.check_position_available(checkers_type, new_pos):
                        new_dice = list(_dice)
                        new_dice.remove(die)

                        new_positions = {*from_positions, new_pos} if new_pos < len(self.board.cols) else from_positions

                        sub_moves = _get_moves(new_dice, new_positions)
                        if new_dice:
                            _result = itertools.product([(pos, die)], sub_moves)
                            yield from (tuple(sorted(_flatten(move), key=lambda m: m[0])) for move in _result)
                        else:
                            yield ((pos, die),)

        positions = list(self.board.get_occupied_positions(checkers_type))
        is_double = (dice[0] == dice[1])

        # Simple check, that all checkers are on the head (First move). In this case we can get two checkers from head.
        first_move = len(positions) == 1 and positions[0] == 0

        # If it is double dice, than we can make 4 moves.
        if is_double:
            dice = dice * 2

        # We check situations, when we can not make move with full dice. In this cases we find other available moves.
        # e.x. We have 6:6. See 6:6:6:6 -> 6:6:6 -> 6:6 -> 6
        # e.x. We have 4:5. See 4:5 -> 4 -> 5
        for used_dice_count in range(len(dice), 0, -1):
            # Find all available combinations of dice.
            available_dice_combs = set(itertools.combinations(dice, used_dice_count))

            _moves = (set(_get_moves(dice_comb, set(positions))) for dice_comb in available_dice_combs)
            # _moves = itertools.chain(*_moves)
            _moves = set(itertools.chain(*_moves))
            available_moves = {moves for moves in _moves if self.board.check_move_available(*moves)}

            if available_moves:
                # In the first step we can get two checker from the head if it is double dice
                if is_double and first_move:
                    die = dice[0]
                    if self.board.check_position_available(checkers_type, 0 + die):
                        available_moves.add(((0, die), (0, die)))

                return available_moves

        return set()


if __name__ == '__main__':
    # board = Board()
    players = [RandomPlayer(), RandomPlayer()]
    game = Game(players=players, logs=True)
    game.play()


