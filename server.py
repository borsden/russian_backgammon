from enum import Enum
from typing import Set

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, root_validator
from pydantic import Field
from starlette.middleware.cors import CORSMiddleware

import backgammon.game as bg
from backgammon import agents

app = FastAPI(title="Russian backgammon API.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AgentTypes(Enum):
    """Agent types."""
    random = agents.RandomAgent
    # dqn = agents.NNAgent.with_model_constructor(DQN().load_state_dict('./123'))

    @property
    def agent(self):
        """Receive instance of agent."""
        return AgentTypes[self.name].value()


Agent = Enum('Agent', {i.name: i.name for i in AgentTypes}, type=AgentTypes)


class BoardSchema(BaseModel):
    """Scheme to construct board."""
    columns: bg.ColumnCheckersNumber = Field(
        ...,
        description='Number of checkers in cells for FIRST player. Cell positions start from TOP-RIGHT corner.',
        example={18: 2, 19: 2, 20: 2, 21: 2, 22: 1, 23: 1}
    )
    opponent_columns: bg.ColumnCheckersNumber = Field(
        ...,
        description='Number of checkers in cells for SECOND player. Cell positions start from BOTTOM-LEFT corner.',
        example={16: 3, 20: 6, 21: 6}
    )

    @property
    def board(self):
        """Create Board instance form specified columns with checkers."""
        board = bg.Board.from_schema(self.columns, self.opponent_columns)
        return board

    @root_validator
    def check_board_available(cls, values):
        """Try to create board from specified parameters."""
        columns, opponent_columns = values.get('columns'), values.get('opponent_columns')
        board = bg.Board.from_schema(columns, opponent_columns)
        return values


class MoveSchema(BaseModel):
    """Scheme with board and movement direction."""
    board: BoardSchema
    reverse: bool = Field(
        False,
        description='Flag, that direction of move is reversed. True for SECOND player.',
        example=False
    )


class ResponseOutput(BaseModel):
    """Scheme to return board position and random dice."""
    board: BoardSchema
    dice: bg.Dice = Field(..., description='Dice.', example=(2, 4))
    moves: bg.Moves = Field(tuple(), description='Made moves. Move - (Start position, Step)', example=((3, 5), (7, 11)))
    available_moves: Set[bg.Moves] = Field(
        {},
        description='Available moves. Move - (Start position, Step)',
        example={((3, 7), (7, 9)), ((3, 5), (7, 11))}
    )


@app.post("/agent_move/{agent}", response_model=ResponseOutput, response_model_exclude={"available_moves"})
async def agent_move(agent: Agent, schema: MoveSchema):
    """Endpoint to receive a move with random dice for specified agent on current board position."""
    try:
        board = schema.board.board
        agent = agent.agent
        with board.reverse(fake=not schema.reverse) as board:
            dice = bg.roll_dice()
            available_moves = board.get_available_moves(dice)
            moves = tuple()
            if available_moves:
                moves = agent.get_action(available_moves, board)
                board.move(*moves)
        columns, opponent_columns = board.to_schema()
        return dict(
            board=dict(columns=columns, opponent_columns=opponent_columns),
            moves=moves,
            dice=dice
        )
    except Exception as e:
        print(e)
        raise HTTPException(404, "Specified board is not correct.")


@app.post("/roll_dice", response_model=ResponseOutput, response_model_exclude={"moves"})
async def roll_dice(schema: MoveSchema):
    """Endpoint to roll dice and receive available moves for them."""
    try:
        board = schema.board.board
        with board.reverse(fake=not schema.reverse) as board:
            dice = bg.roll_dice()
            available_moves = board.get_available_moves(dice)
        columns, opponent_columns = board.to_schema()
        return dict(
            board=dict(columns=columns, opponent_columns=opponent_columns),
            available_moves=available_moves,
            dice=dice
        )
    except Exception as e:
        print(e)
        raise HTTPException(404, "Specified board is not correct.")


@app.get("/available_agents", response_model=Set[str])
async def available_agents():
    """Endpoint to receive available agents."""
    return {i.name for i in AgentTypes}
