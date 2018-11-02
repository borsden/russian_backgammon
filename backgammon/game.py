import abc
import copy
import itertools
import random
from contextlib import contextmanager
from typing import List, Tuple, NamedTuple, Sequence, Set, Iterator, Any, Optional


class MoveError(Exception):
    pass


class Agent:
    """Player."""
    checker_type: 'Checker'
    """Checker type for this player."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abc.abstractmethod
    def get_action(self, available_moves: Set['Moves'], game: 'Game') -> 'Moves':
        """Strategy of player."""


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
    CHECKER_TYPES = [xChecker, oChecker]

    def __init__(self) -> None:
        self.cols = [[] for _ in range(self.NUM_COLS)]
        """Columns of a board."""

        self.cols[0] = [self.CHECKER_TYPES[0] for _ in range(self.NUM_CHECKERS)]
        self.cols[self.half_cols_len] = [self.CHECKER_TYPES[1] for _ in range(self.NUM_CHECKERS)]
        """Add checkers at the heads."""

    @property
    def current_checker(self):
        """Current checker."""
        return self.CHECKER_TYPES[0]

    @property
    def opponent_checker(self):
        """Opponent checker."""
        return self.CHECKER_TYPES[1]

    def get_winner_checker_type(self) -> Checker:
        """Get winner checker type if exist."""
        if self.was_finished():
            return next(checker_type for checker_type in self.CHECKER_TYPES if self.is_winner(checker_type))

    def get_occupied_positions(self, opponent: bool = False) -> Iterator[int]:
        """Get positions occupied by current checker type.

        :param opponent: find occupied positions for opponent
        """
        checker_type = self.opponent_checker if opponent else self.current_checker
        return (index for index, col in enumerate(self.cols) if col and col[0] == checker_type)

    def check_position_available(self, position: int) -> bool:
        """Check, that this position is available for this checker type.
        Also available positions out of board. It means checker withdrawal.

        :param position: position
        """
        if position < len(self.cols):
            col = self.cols[position]
            return not col or col[0] == self.current_checker
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
        if checker_type == self.current_checker:
            yield self
        else:
            with self.reverse():
                yield self

    @contextmanager
    def reverse(self) -> 'Board':
        """Context manager to turn board and return after."""
        def _reverse():
            self.CHECKER_TYPES[0], self.CHECKER_TYPES[1] = self.CHECKER_TYPES[1], self.CHECKER_TYPES[0]
            self.cols[half_len:], self.cols[:half_len] = self.cols[:half_len], self.cols[half_len:]

        half_len = self.half_cols_len
        _reverse()
        yield self
        _reverse()

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
                    prev_positions = [pos for pos in self.get_occupied_positions() if pos < from_pos]
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
            _move(move)

        if not self.can_make_blocks() and self.has_block():
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

    def was_finished(self) -> bool:
        """Game was finished if one of players is winner."""
        return any(
            self.is_winner(checker_type)
            for checker_type in self.CHECKER_TYPES
        )

    def is_winner(self, checker_type: Checker)-> bool:
        """Check, that current checker_type does not have checkers on board."""
        with self.viewpoint(checker_type):
            return not list(self.get_occupied_positions())

    def made_mars(self, checker_type: Checker)-> bool:
        """Check, that this checker type made mars for opponent.

        (Was finished, but all opponent checker are on board.)
        """
        with self.viewpoint(checker_type):
            opponents_positions = self.get_occupied_positions(opponent=True)

            return (
                self.is_winner(checker_type) and
                sum(
                    len(self.cols[pos])
                    for pos in opponents_positions
                ) == self.NUM_CHECKERS
            )

    def made_koks(self, checker_type: Checker)-> bool:
        """Check, that this checker type made koks for opponent.

        (Was finished, and one or more checkers of opponent are in first quadrant.)
        """
        with self.viewpoint(checker_type):
            opponents_positions = self.get_occupied_positions(opponent=True)
            opponent_home = (self.NUM_COLS // 2, 3 * self.NUM_COLS // 4)

            return (
                self.is_winner(checker_type) and
                any(
                    opponent_home[0] <= pos < opponent_home[1]
                    for pos in opponents_positions
                )
            )

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
        return not any(pos < home_pos for pos in self.get_occupied_positions() )

    def check_correct(self):
        """Simple check that board is correct.
         It means, that we have 15 checkers for each type of checkers (except situations, when all checkers are in home).
        """
        for checker_type in self.CHECKER_TYPES:
            with self.viewpoint(checker_type):
                if not self.can_withdraw(checker_type):
                    positions = self.get_occupied_positions()
                    total_checkers = sum(len(self.cols[pos]) for pos in positions)
                    if total_checkers != self.NUM_CHECKERS:
                        return False
        return True


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

        for player, checker_type in zip(self.players, self.board.CHECKER_TYPES):
            player.checker_type = checker_type

        who_start = random.randint(0, 1) if who_start is None else who_start
        self._store['who_start'] = who_start

        self.players_steps = itertools.cycle(self.players if not who_start else reversed(self.players))
        """Infinitive iterator for order of player steps. Starting player is random."""

    def play(self) -> Agent:
        """Play game. Return winner after end."""

        while not self.board.was_finished():
            current_player = next(self.players_steps)
            self.make_step(player=current_player)

        return current_player

    @property
    def is_end(self):
        """Check, that game was end."""
        # raise NotImplementedError
        return False

    def print(self, *values: Any) -> None:
        """Print value if logs flag is enabled."""
        if self.show_logs:
            print(values)

    def make_step(self, player: Agent) -> None:
        """Roll dice, get available moves for them and get action from player.

        :param player: current Player.
        """
        dice = self.roll_dice()
        self._store['dice'].append(dice)


        if self.show_logs:
            self.board.draw()
        with self.board.viewpoint(player.checker_type):

            available_moves = self.get_available_moves(dice)

            self.print('Dice:', dice)
            self.print('Available moves:', available_moves)

            if available_moves:
                moves = player.get_action(available_moves, self)
                self._store['moves'].append(moves)
                self.board.move(*moves)

                self.print('Move:', moves)

    def roll_dice(self) -> Dice:
        """Roll dice and return their values."""
        return random.randint(1, 6), random.randint(1, 6)

    def get_available_moves(self, dice: Dice) -> Set[Moves]:
        """Return available moves for current player with that dices.

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

                    if self.board.check_position_available(new_pos):
                        new_dice = list(_dice)
                        new_dice.remove(die)

                        if new_dice:
                            new_positions = {*from_positions, new_pos} if new_pos < len(
                                self.board.cols) else from_positions

                            yield from (
                                tuple(sorted(current_moves + sub_move, key=lambda m: m[0]))
                                for sub_move in _get_moves(new_dice, new_positions)
                            )
                        else:
                            yield current_moves

        positions = list(self.board.get_occupied_positions())
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
            available_moves = {moves for moves in _moves if self.board.check_move_available(*moves)}

            if available_moves:
                # In the first step we can get two checker from the head if it is double dice
                if is_double and first_move:
                    die = dice[0]
                    if self.board.check_position_available(0 + die):
                        available_moves.add(((0, die), (0, die)))

                return available_moves

        return set()

