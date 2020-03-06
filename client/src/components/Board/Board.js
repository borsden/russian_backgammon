import React from 'react';
import './Board.css';
import DiceArea from '../DiceArea/DiceArea';
import Triangle from './Triangle/Triangle';
import Checker from "../Checker/Checker";

const
    BottomLeftPositions = [...Array(6)].map((_, index) => [12 + index, index]),
    BottomRightPositions = BottomLeftPositions.map(([index, oppIndex]) => [index + 6, oppIndex + 6]),
    TopLeftPositions = [...BottomRightPositions].reverse().map(([index, oppIndex]) => [oppIndex, index]),
    TopRightPositions = [...BottomLeftPositions].reverse().map(([index, oppIndex]) => [oppIndex, index]);

const getCheckers = (player, numberOfCheckers, highlighted = false) => {
    if (player && numberOfCheckers) {
        //mount up to 5 checkers
        const count = numberOfCheckers > 5 ? 5 : numberOfCheckers;
        //checkers array
        const checkers = [];

        //Get checkers
        for (let i = 0; i < count; i++) {
            //highlight last checker if it can move
            const key = `board${player}P${i}`;
            if (highlighted && i === count - 1) {
                checkers.push(<Checker player={player} count={1} key={key} highlighted={1}/>);
            } else {
                checkers.push(<Checker player={player} count={1} key={key}/>);
            }
        }

        //add label to the first checker if the point has more than 5 checkers
        if (numberOfCheckers > 5) {
            checkers[0] = <Checker player={player} count={numberOfCheckers - 4} key={`board${player}P0`}/>;
        }

        return checkers
    } else {
        return null;
    }
};

const block = (positions, columns, opponentColumns, canMove, canReceive, reverseMove, bottom) => (
    <div className={`${bottom ? 'blocksDown' : 'blocksUp'}`}>
        {
            positions.map(([pos, oppPos], index) => (
                <Triangle
                    otherColor={index % 2 !== 0}
                    position={`${bottom ? 'bottom' : 'top'}`}
                    key={index}
                    canMove={reverseMove ? canMove[oppPos] : canMove[pos]}
                    canReceive={reverseMove ? canReceive[oppPos] : canReceive[pos]}
                >
                    {getCheckers(1, columns[pos], !reverseMove && !!canMove[pos])}
                    {getCheckers(2, opponentColumns[oppPos], reverseMove && !!canMove[oppPos])}
                </Triangle>
            ))
        }
    </div>
);

const board = ({opponentColumns, columns, reverseMove, rollDice, dice, children, isRunning, canMove, canReceive}) => {
    let leftDiceArea = null;
    let rightDiceArea = null;
    if (isRunning) {
        leftDiceArea = reverseMove && <DiceArea dice={dice} clicked={rollDice}/>;
        rightDiceArea = (!reverseMove) && <DiceArea dice={dice} clicked={rollDice}/>;
    }

    return (
        <div id="board" className="container-fluid">
            <div id="board" className="container-fluid">

                <div id="leftSide" className="row">
                    {leftDiceArea}
                    {block(TopLeftPositions, columns, opponentColumns, canMove, canReceive, reverseMove, false)}
                    {block(BottomLeftPositions, columns, opponentColumns, canMove, canReceive, reverseMove, true)}
                </div>
                <div id="rightSide" className="row">
                    {rightDiceArea}
                    {block(TopRightPositions, columns, opponentColumns, canMove, canReceive, reverseMove, false)}
                    {block(BottomRightPositions, columns, opponentColumns, canMove, canReceive, reverseMove, true)}
                </div>
            </div>
            {children}
        </div>
    );

};

export default board;
