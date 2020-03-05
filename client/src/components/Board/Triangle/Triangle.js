import React from 'react';
import './Triangle.css';

const Triangle = ({position, otherColor, canReceive, canMove, children}) => {

    let classOrientation = '';
    let classColor = '';
    let classReceivable = '';
    let pointContainerClasses = '';


    if (position === "top") {
        classOrientation = "Up";
    }
    else {
        classOrientation = "Down";
        pointContainerClasses = " pointContainerDown";
    }

    if (otherColor) {
        classColor += "C2";
    }

    let action = null;
    if (canReceive) {
        action = canReceive;
        pointContainerClasses += ' containerClickable';
        classReceivable = 'Receivable';
        classColor = '';
    }
    if (canMove) {
        action = canMove;
        pointContainerClasses += ' containerClickable';
    }

    return (
        <div className="triangle col-xs-2 " >
            <div className={"trianglePart triangleLeft" + classOrientation + classColor + classReceivable}/>
            <div className={"trianglePart triangleRight" + classOrientation + classColor + classReceivable}/>
            <div className={"pointContainer " + pointContainerClasses} onClick={action}>
                {children}
            </div>
        </div>

    );
};

export default Triangle;