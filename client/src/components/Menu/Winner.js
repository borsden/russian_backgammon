import React, {Component} from 'react';
import './Menu.css';


class Winner extends Component {

    render() {

        return (
            <div id="modal">

                <div id="modal-content" className="menu-small">
                    <div id="modal-body">
                        <img src="./assets/congratulation.png" alt="congratulation"/>
                        <p className="modal-body-centralized">{this.props.winner} wins!</p>
                    </div>
                    <div id="modal-footer">
                        <button
                            className="btn btn-success"
                            onClick={this.props.closeGameHandler}>New Game
                        </button>
                    </div>
                </div>

            </div>
        );
    }


}

export default Winner;