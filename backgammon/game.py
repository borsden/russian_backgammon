import abc
import copy
import itertools
import random
from contextlib import contextmanager
from typing import List, Tuple, NamedTuple, Sequence, Set, Iterator, Any, Optional, Dict


class MoveError(Exception):
    pass


class Agent:
    """Player."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abc.abstractmethod
    def get_action(self, available_moves: Set['Moves'], board: 'Board') -> 'Moves':
        """Strategy of player."""


class Checker(NamedTuple):
    """Checker."""
    type: str

    def __str__(self):
        return self.type


ColumnCheckersNumber = Dict[int, int]
"""Number of column and number of checkers in this column."""

Dice = Tuple[int, int]
"""Roll of the dice."""

Move = Tuple[int, int]
"""Move on a board. Include start position and die."""

Moves = Tuple[Move, ...]
"""Moves in one step."""


def roll_dice() -> Dice:
    """Roll dice and return their values."""
    return random.randint(1, 6), random.randint(1, 6)


class Column(NamedTuple):
    """Column of a board."""
    is_opp: Optional[bool] = None
    """Indicate type of checkers in column. True for opponent (opponent by default), False for player."""
    checkers: int = 0
    """Number of checkers into column."""


class Board:
    """Backgammon board."""
    NUM_COLS = 24
    """Number of columns."""
    NUM_CHECKERS = 15
    """Number of checkers for one player."""

    def __init__(self, empty=False) -> None:
        # Columns of a board.
        self.columns: Dict[int, Column] = {
            column: Column() for column in range(self.NUM_COLS)
        }

        if not empty:
            # Add checkers at the heads.
            self.columns[0] = Column(False, self.NUM_CHECKERS)
            self.columns[self.NUM_COLS // 2] = Column(True, self.NUM_CHECKERS)

        # Flag to check, that board is reversed
        self._is_opp = False

    @classmethod
    def from_schema(
            cls,
            x_columns: ColumnCheckersNumber,
            y_columns: ColumnCheckersNumber,
            straight: bool = False
    ) -> 'Board':
        """Method to create board from passed columns and number of checkers there.

        :param x_columns: checkers in columns for FIRST player
        :param y_columns: checkers in columns for SECOND player
        :param straight: Flag, that columns positions for second player started from first player view
                (0->23, 2->21, ...)
        """

        def fill_board(board: Board, columns: ColumnCheckersNumber, is_opp: bool) -> None:
            for pos, checkers in columns.items():
                if board.columns[pos].checkers:
                    raise ValueError(f'Columns should not include both types of checkers.')

                board.columns[pos] = Column(is_opp, checkers)

        board = cls(empty=True)

        fill_board(board, x_columns, False)

        with board.reverse(fake=straight) as board:
            fill_board(board, y_columns, True)

        board.check_correct()
        return board

    def to_schema(self, straight: bool = False) -> Tuple[ColumnCheckersNumber, ColumnCheckersNumber]:
        """Method to receive columns numbers and number of checkers there from existing board

        :param straight: Flag, that columns positions for second player should started from first player view
                (0->23, 2->21, ...)
        """
        positions = self.get_occupied_positions()
        column_checker = {pos: self.columns[pos].checkers for pos in positions}

        with self.reverse(fake=straight) as board:
            opponent_positions = board.get_occupied_positions(opponent=straight)
            opponent_column_checker = {pos: board.columns[pos].checkers for pos in opponent_positions}
        return column_checker, opponent_column_checker

    def is_winner(self, is_opp: bool) -> bool:
        """Check, that current checker_type does not have checkers on board."""
        with self.reverse(fake=not is_opp):
            return not list(self.get_occupied_positions())

    def made_mars(self, is_opp: bool) -> bool:
        """Check, that this checker type made mars for opponent.

        (Was finished, but all opponent checker are on board.)
        """
        with self.reverse(fake=not is_opp):
            opponents_positions = self.get_occupied_positions(opponent=True)

        return (
                self.is_winner(is_opp) and
                sum(
                    len(self.columns[pos])
                    for pos in opponents_positions
                ) == self.NUM_CHECKERS
        )

    def made_koks(self, is_opp: bool) -> bool:
        """Check, that this checker type made koks for opponent.

        (Was finished, and one or more checkers of opponent are in first quadrant.)
        """
        with self.reverse(fake=not is_opp):
            opponents_positions = self.get_occupied_positions(opponent=True)
            opponent_home = (self.NUM_COLS // 2, 3 * self.NUM_COLS // 4)

        return (
                self.is_winner(is_opp) and
                any(
                    opponent_home[0] <= pos < opponent_home[1]
                    for pos in opponents_positions
                )
        )

    @property
    def status(self) -> Optional[int]:
        """Status of a game.

        None, if game is not finished 1/-1 if current player win/lose, 2/-2 for mars, 3/-3 for koks.
        """

        def _status(is_opp):
            K = -1 if is_opp else 1
            with self.reverse(fake=not is_opp):
                finished = not list(self.get_occupied_positions())
                # Check, if player don't have any checkers on a board.
                if finished:
                    opp_positions = list(self.get_occupied_positions(opponent=True))
                    # Check, if opponent have all checkers on a board. (MARS)
                    if sum(self.columns[pos].checkers for pos in opp_positions) == self.NUM_CHECKERS:
                        opponent_home = (self.NUM_COLS // 2, 3 * self.NUM_COLS // 4)
                        # Check, if one or more checkers of opponent are in first quadrant.
                        if any(opponent_home[0] <= pos < opponent_home[1] for pos in opp_positions):
                            return K * 3
                        return K * 2

                    return K

        return _status(False) or _status(True)

    def get_occupied_positions(self, opponent: bool = False) -> Iterator[int]:
        """Get occupied positions.

        :param opponent: find occupied positions for opponent
        """
        return (
            pos for pos, col in self.columns.items()
            if col.checkers and
               opponent != (col.is_opp == self._is_opp)
        )

    def check_position_available(self, position: int) -> bool:
        """Check, that this position is available.
        Also available positions out of board. It means checker withdrawal.

        :param position: position
        """
        if position < self.NUM_COLS:
            column = self.columns[position]
            return not column.checkers or column.is_opp == self._is_opp
        else:
            return True

    def print(self) -> None:
        """Draw current board."""

        def half_row() -> List[List[str]]:
            """Make quadrants of board with current position."""
            max_length = max(self.columns[pos].checkers for pos in range(half_cols_len))

            rows = [
                [
                    (' o' if self.columns[pos].is_opp else ' x') if self.columns[pos].checkers > checker_pos else '  '
                    for pos in range(half_cols_len)
                ]
                for checker_pos in range(max_length)
            ]
            return rows

        def col_pos(second_half: bool = False) -> List[str]:
            """Column position. """
            return [str(col + (half_cols_len if second_half else 0)).rjust(2, ' ') for col in range(half_cols_len)]

        def horizontal_delimiter():
            """Simple separate positions and checkers."""
            return ['--' for _ in range(half_cols_len)]

        half_cols_len = self.NUM_COLS // 2

        rows = [
            col_pos(),
            horizontal_delimiter(),
            *half_row(),
            [' _' for _ in range(half_cols_len)],
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
    def reverse(self, fake: bool = False) -> 'Board':
        """Context manager to turn board and return previous view after that.

        :param fake: don't turn board, return as is.
        """

        def _reverse():
            self.columns = {
                ((pos + half_cols_len) if pos < half_cols_len else (pos - half_cols_len)): col
                for pos, col in self.columns.items()
            }
            self._is_opp = not self._is_opp

        half_cols_len = self.NUM_COLS // 2
        if fake:
            yield self
        else:
            _reverse()
            yield self
            _reverse()

    def move(self, *moves: Move) -> None:
        """Move checkers.

        May multiple moves.
        """

        def _move(move: Move):
            """Move one checker from position with step."""
            from_pos, die = move
            to_pos = from_pos + die
            if not self.columns[from_pos]:
                raise MoveError('`From` position is empty.')

            from_col = self.columns[from_pos]
            if from_col.is_opp != self._is_opp:
                raise MoveError('`From` position is not player position.')

            if to_pos < self.NUM_COLS:
                new_col = self.columns[to_pos]
                if new_col.checkers and new_col.is_opp != from_col.is_opp:
                    raise MoveError('`To` position is already opponent position.')
                self.columns[to_pos] = Column(from_col.is_opp, new_col.checkers + 1)
            else:
                if not self.can_withdraw():
                    raise MoveError('Tried to withdraw checker when not all checkers in the home.')
                if to_pos != self.NUM_COLS:
                    prev_positions = [pos for pos in self.get_occupied_positions() if pos < from_pos]
                    if prev_positions:
                        raise MoveError('Previous checkers can be withdraw.')

            # Todo: is it better to set is_opp=None if resulted checkers=0
            from_checkers = from_col.checkers - 1
            self.columns[from_pos] = Column(from_col.is_opp if from_checkers else None, from_checkers)

        def _rollback(*moves: Move):
            """Rollback all moves. Simple use `to` position and negative step."""
            rollback_moves = ((move[0] + move[1], -move[1]) for move in reversed(moves))
            for move in rollback_moves:
                _move(move)

        if not moves:
            raise MoveError('Must be at least one move in args.')

        for move in moves:
            _move(move)

        if not self.can_make_blocks() and self.has_block():
            _rollback(*moves)
            raise MoveError('Block is unavailable while opponent has not at least one checker in home.')

    @contextmanager
    def temp_move(self, *moves: Move) -> 'Board':
        """Temp move checkers contextmanager. After we return checkers, where they were."""
        columns = dict(self.columns)
        try:
            self.move(*moves)
            yield self
        except Exception as e:
            raise e
        finally:
            self.columns = columns

    def has_block(self, block_size: int = 6) -> bool:
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
            for block in possible_blocks(self.get_occupied_positions())
        )

    def can_make_blocks(self) -> bool:
        """Check that blocks is available.

        If opponent has at least one checker in his home, than available.
        """
        opponent_home = (self.NUM_COLS // 4, self.NUM_COLS // 2)

        opponents_positions = self.get_occupied_positions(opponent=True)

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

    def can_withdraw(self) -> bool:
        """Check, that all checkers are in home."""
        home_pos = 3 * self.NUM_COLS // 4
        return not any(pos < home_pos for pos in self.get_occupied_positions())

    def check_correct(self):
        """Simple check that board is correct.
         It means, that we have 15 checkers for each type of checkers (except situations, when all checkers are in home).
        """
        for fake in [True, False]:
            with self.reverse(fake=fake):
                if not self.can_withdraw():
                    positions = self.get_occupied_positions()
                    total_checkers = sum(self.columns[pos].checkers for pos in positions)
                    if total_checkers != self.NUM_CHECKERS:
                        raise ValueError('Incorrect board. Should heave 15 checkers for each type of checkers.')
                if not self.can_make_blocks() and self.has_block():
                    raise ValueError('Can block only if at least one opponent checker is in home.')

    def get_available_moves(self, dice: Dice) -> Set[Moves]:
        """Return available moves with specified dice.

        :param dice: dice
        """

        def _get_moves(_dice: Sequence[int], from_positions: Set[int]) -> Set[Moves]:
            """Find all available moves.
            Iterate for every available position and every die in dice.
            If step with this die from this position is available, then we add this position to existing positions
            and recursive check other positions and dice.

            :param _dice: checking dice
            :param from_positions: available positions for steps
            """
            for pos in from_positions:
                for die in _dice:
                    new_pos = pos + die
                    current_moves = ((pos, die),)

                    if self.check_position_available(new_pos):
                        new_dice = list(_dice)
                        new_dice.remove(die)

                        if new_dice:
                            new_positions = {new_pos for new_pos in from_positions if new_pos >= pos}
                            if new_pos < len(self.columns):
                                new_positions.add(new_pos)

                            yield from (
                                tuple(current_moves + sub_move)
                                for sub_move in _get_moves(new_dice, new_positions)
                            )
                        else:
                            yield current_moves

        # dice = sorted(dice)
        positions = list(self.get_occupied_positions())
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
            _moves = set(itertools.chain(*_moves))
            available_moves = {moves for moves in _moves if self.check_move_available(*moves)}

            if available_moves:
                # In the first step we can get two checker from the head if it is double dice
                if is_double and first_move:
                    die = dice[0]
                    if self.check_position_available(0 + die):
                        double_move = ((0, die), (0, die))
                        # Resolve bug, when we have ((0,6)) and ((0,6), (0,6)) moves
                        if die == 6:
                            return {double_move}

                        available_moves.add(double_move)

                return available_moves

        return set()


class Game:
    """Game. """

    def __init__(self, players: List[Agent], show_logs: bool = False, who_start: Optional[int] = None) -> None:
        """
        :param players: agents, who play this game
        :param show_logs: do we want logout
        :param who_start: manual set starting player
        """
        self.show_logs = show_logs
        self._store = dict(
            dice=[],
            moves=[],
            who_start=0
        )

        self.board = Board()
        """Board."""
        self.players = players
        """Players of current game."""

        who_start = random.randint(0, 1) if who_start is None else who_start
        first_player_start = not bool(who_start)

        self._store['who_start'] = who_start

        self.players_steps = itertools.cycle(self.players if first_player_start else reversed(self.players))
        """Infinitive iterator for order of player steps. Starting player is random."""

    def play(self) -> int:
        """Play game. Return board status after end."""
        is_opp = False
        while self.board.status is None:
            current_player = next(self.players_steps)
            self.make_step(player=current_player, is_opp=is_opp)
            is_opp = not is_opp
        return self.board.status

    def print(self, *values: Any) -> None:
        """Print value if logs flag is enabled."""
        if self.show_logs:
            print(values)

    def make_step(self, player: Agent, is_opp: bool) -> None:
        """Roll dice, get available moves for them and get action from player.

        :param player: current Player.
        :param is_opp: if player sit on opponent seat
        """
        dice = roll_dice()
        self._store['dice'].append(dice)

        if self.show_logs:
            self.board.print()
        with self.board.reverse(fake=not is_opp):

            available_moves = self.board.get_available_moves(dice)

            self.print('Dice:', dice)
            self.print('Available moves:', available_moves)

            if available_moves:
                moves = player.get_action(available_moves, self.board)
                self._store['moves'].append(moves)
                self.print('Move:', moves)

                self.board.move(*moves)

                self.print('Move:', moves)
