import React from 'react';
import './OutSideBar.css';

import CheckerFlat from './CheckerFlat/CheckerFlat';

const OutSideBar = ({undoHandler, checkers, canReceive, reverseMove, opponentCheckers}) => {


    const getFlatCheckers = (player, outedCheckers) => {

        // const checkers = [...Array(15 - Object.values(cols).reduce((a,b) => a + b, 0))].map(
        const checkers = [...Array(15 - outedCheckers)].map(
            (_, i) => <CheckerFlat player={player} key={`OSBP${player}C${i}`} />
        );
        return checkers
    }


    const checkersP1 = getFlatCheckers("1", checkers);
    const checkersP2 = getFlatCheckers("2", opponentCheckers);

    let undoButtonclass = 'disabled';
    let undoButtonFunction;
    if (undoHandler) {
        undoButtonclass = '';
        undoButtonFunction = undoHandler;
    }


    return (
        <div id="outSide" className="row">

            <div className="undoButton">
                <button
                    className={`btn btn-warning ${undoButtonclass}`}
                    onClick={undoButtonFunction}>Undo
                    </button>
            </div>

            <div className="blocksUp" onClick={(!reverseMove && canReceive) ? canReceive: null}>
                <div className={`pointContainer ${(canReceive && !reverseMove) ? 'receivable': ''}`}>
                    {checkersP1}
                </div>
            </div>

            <div className="blocksDown" onClick={(reverseMove && canReceive) ? canReceive: null}>
                <div className={`pointContainer pointContainerDown ${(canReceive && reverseMove) ? 'receivable': ''}`}>
                    {checkersP2}
                </div>
            </div>
        </div>
    )
}

export default OutSideBar;