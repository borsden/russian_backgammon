import React from 'react';
import './DiceArea.css';

import Dice from './Dice/Dice';
import RollButton from './RollButton/RollButton';

const diceArea = ({dice, clicked}) => {
    let area = '';
    if (!dice) {
        area = <RollButton label="Roll Dice" clicked={clicked}/>
    } else {
        area = dice.map((number, index) => (<Dice diceNumber="index" number={number} key={`dice${index}`}/>));
    }

    return (
        <div className="diceArea">
            {area}
            {/*{(dice && !canMove) ? <RollButton label="No Moves available"/> : null}*/}
        </div>
    )
};

export default diceArea