import React, {Component} from 'react';
import './Menu.css';


class Menu extends Component {
    state = {
        player1: undefined,
        player2: undefined,
    };

    newGameHandler = () => {
        this.props.newGameHandler([
            this.state.player1 || this.props.availablePlayers[0],
            this.state.player2 || this.props.availablePlayers[0]
        ])
    };

    render() {
        return (
            <div id="modal">

                <div id="modal-content" className="menu-medium">
                    <div id="modal-body">
                        <div className="modal-body-centralized">RL Russian backgammon</div>
                    </div>
                    <div id="modal-footer">

                        <select value={this.state.player1} id="select-1-player"
                                onChange={(event) => this.setState({player1: event.target.value})}>
                            {this.props.availablePlayers.map((player) => (
                                    <option value={player} key={player}>{player.toUpperCase()}</option>
                                )
                            )}
                        </select>
                        <select value={this.state.player2} id="select-2-player"
                                onChange={(event) => this.setState({player2: event.target.value})}>
                            {this.props.availablePlayers.map((player) => (
                                    <option value={player} key={player}>{player.toUpperCase()}</option>
                                )
                            )}
                        </select>
                        <button
                            className="btn btn-success"
                            onClick={this.newGameHandler}>New Game
                        </button>
                    </div>
                </div>

            </div>
        );
    }
}

export default Menu;