import React, {Component} from 'react';
import './App.css';

import Board from '../components/Board/Board';
import OutSideBar from "../components/OutSideBar/OutSideBar";
import Menu from '../components/Menu/Menu';
import Winner from '../components/Menu/Winner';

const SERVER_URL = 'http://localhost:8000';
const sleep = (duration) => new Promise(resolve => setTimeout(resolve, duration));

class App extends Component {
    //Initial state
    state = {
        columns: {0: 15},
        opponentColumns: {0: 15},

        history: [],

        availableMoves: null,
        moves: [],
        selectedColumn: null,

        reverseMove: Math.random() >= 0.5,

        agentMove: null, // Current agent move
        players: [],

        availablePlayers: [],
        sleep: 1000,
    };
    async componentDidMount() {
        this.setState({availablePlayers: await this.receiveAvailablePlayers()})
    }

    get currentPlayer() {
        return this.state.reverseMove ? this.state.players[1] : this.state.players[0]
    }

    get isPlayer() {
        return this.currentPlayer === 'player'
    }

    receiveAvailablePlayers = async () => {
        try {
        const
            response = await fetch(`${SERVER_URL}/available_agents`),
            availableAgents = await response.json();
            return ['player', ...availableAgents]
        }
        catch (e) {
            alert('Could not receive available agents. Will reload.');
            window.location.reload();
        }

    };

    closeGameHandler = () => {
      this.setState({
              columns: {0: 15},
              opponentColumns: {0: 15},
              history: [],

              availableMoves: null,
              moves: [],
              selectedColumn: null,
              reverseMove: Math.random() >= 0.5,
              agentMove: null,
              players: [],
      })
    };

    newGameHandler = (players) => {
        this.setState({
            players
        })
    };

    postServer = async () => {
        const
            reverse = this.state.reverseMove,
            currentPlayer = this.currentPlayer,
            endpoint = this.isPlayer ? 'roll_dice' : `agent_move/${currentPlayer.toLowerCase()}`;

        const response = await fetch(`${SERVER_URL}/${endpoint}`, {
            method: 'POST', // или 'PUT'
            body: JSON.stringify({
                board: {
                    columns: this.state.columns,
                    opponent_columns: this.state.opponentColumns,
                },
                reverse
            }),
            headers: {
                'Content-Type': 'application/json'
            }
        });
        return await response.json()
    };

    moveAgent = async ({moves}) => {
        if (!moves.length) {
            await sleep(this.state.sleep);
        }
        for (let [pos, step] of moves) {
            this.setState({agentMove: [pos, step]});
            await sleep(2 * this.state.sleep);
            let newState = {
                columns: {...this.state.columns},
                opponentColumns: {...this.state.opponentColumns},
                history: [this.state.history, this.state]
            };
            let currentColumns = this.state.reverseMove ? newState.opponentColumns : newState.columns;

            currentColumns[pos] -= 1;
            if (!currentColumns[pos]) {
                delete currentColumns[pos]
            }

            const nextPos = pos + step;
            if (nextPos < 24) {
                currentColumns[nextPos] = (currentColumns[nextPos] || 0) + 1;
            }

            this.setState(newState)
        }
        this.setState({
            dice: null,
            agentMove: null,
            reverseMove: !this.state.reverseMove,
        });
    };


    rollDiceHandler = async () => {
        try {
            const {board, dice, moves, available_moves} = await this.postServer();
            this.setState({dice});
            if (this.currentPlayer !== 'player') {
                this.moveAgent({moves})
            } else {
                if (!available_moves.length) {
                    await sleep(this.state.sleep);
                    this.setState({dice: null, reverseMove: !this.state.reverseMove});
                } else {
                    this.setState({availableMoves: available_moves})
                }
            }
        } catch (error) {
            console.error('Ошибка:', error);
        }

    };
    undoHandler = () => {
        if (this.state.selectedColumn) {
            this.setState({selectedColumn: null})
        } else {
            const prevHistory = this.state.history.slice(-1)[0];
            this.setState({...prevHistory, selectedColumn: null})
        }
    };


    playerColumnSelect = (position) => {
        this.setState({selectedColumn: position})
    };

    playerColumnReceive = (step) => {
        const nextPos = this.state.selectedColumn + step;
        let newState = {

            columns: {...this.state.columns},
            opponentColumns: {...this.state.opponentColumns},
            selectedColumn: null,
            moves: [...this.state.moves, [this.state.selectedColumn, step]],
            history: [this.state.history, this.state]
        };
        let currentColumns = this.state.reverseMove ? newState.opponentColumns : newState.columns;
        const pos = this.state.selectedColumn;
        currentColumns[pos] -= 1;

        if (!currentColumns[pos]) {
            delete currentColumns[pos]
        }

        if (nextPos < 24) {
            currentColumns[nextPos] = (currentColumns[nextPos] || 0) + 1;
        }
        this.setState(newState)

    };

    playerMovesAndReceives = () => {
        let canMove = {}, canReceive = {};
        for (let moves of this.state.availableMoves) {
            moves = moves.map(([position, step]) => `${position}_${step}`);
            let isEqual = true;
            for (let [prevPosition, prevStep] of this.state.moves) {
                let
                    previousMove = `${prevPosition}_${prevStep}`,
                    index = moves.indexOf(previousMove);
                if (index === -1) {
                    isEqual = false;
                    break
                }
                moves.splice(index, 1);
            }
            if (isEqual) {
                if (!moves.length) {
                    this.setState({
                        moves: [],
                        availableMoves: null,
                        dice: null,
                        reverseMove: !this.state.reverseMove,
                    })
                }
                for (let move of moves) {
                    const [position, step] = move.split('_').map(value => parseInt(value));
                    if (this.state.selectedColumn == null) {
                        canMove[position] = () => this.playerColumnSelect(position);
                    }
                    if (position === this.state.selectedColumn) {
                        canReceive[Math.min(position + step, 24)] = () => this.playerColumnReceive(step);
                    }
                }
            }
        }
        return [canMove, canReceive]
    };


    render() {
        const
            agentMove = this.state.agentMove;
        let canMove = {}, canReceive = {};
        if (agentMove) {
            canMove[agentMove[0]] = () => {
            };
            canReceive[Math.min(agentMove[0] + agentMove[1], 24)] = () => {
            }
        }
        if (this.state.availableMoves) {
            [canMove, canReceive] = this.playerMovesAndReceives()
        }

        const
            checkers = Object.values(this.state.columns).reduce((a, b) => a + b, 0),
            opponentCheckers = Object.values(this.state.opponentColumns).reduce((a, b) => a + b, 0),
            winner = (checkers === 0) ? this.state.players[0] : (opponentCheckers === 0 ? this.state.players[1] : null);
        return (
            <div id="App">
                <div id="game">
                    <Board
                        rollDice={this.rollDiceHandler}
                        columns={this.state.columns}
                        opponentColumns={this.state.opponentColumns}
                        reverseMove={this.state.reverseMove}
                        isRunning={true}
                        dice={this.state.dice}
                        canMove={canMove}
                        canReceive={canReceive}
                    />
                    <OutSideBar
                        opponentCheckers={opponentCheckers}
                        checkers={checkers}
                        reverseMove={this.state.reverseMove}
                        canReceive={canReceive[24]}
                        undoHandler={(this.state.selectedColumn || this.state.moves.length > 0) ? this.undoHandler : null}
                    />
                </div>
                {winner && <Winner winner={winner} closeGameHandler={this.closeGameHandler}/>}
                {!this.state.players.length && (
                    <Menu
                        newGameHandler={this.newGameHandler}
                        availablePlayers={this.state.availablePlayers}
                    />
                )}
            </div>

        );
    }
}

export default App;
