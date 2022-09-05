<?php

namespace Rubix\ML\ReinforcementLearning;

/**
 * An action that is floating point between a maximum and a minimum values.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Pablo Duboue
 */
class ContinuousAction implements Action
{
    /**
     * The action space
     *
     * @var ActionType
     */
    protected ActionType $actionSpace;

    /**
     * The value
     *
     * @var float
     */
    protected float $value;

    /**
     * Create a continuous action.
     *
     * @param float $value
     * @param ActionType $actionSpace
     */
    public function __construct(float $value, ActionType $actionSpace)
    {
        $this->value = $value;
        //TODO validate $actionSpace is Continuous
        //TODO validate $value is within range
        $this->actionSpace = $actionSpace;
    }

    /**
     * The selected value.
     *
     * @return float
     */
    public function value() : float
    {
        return $this->value;
    }

    /**
     * Action space this action belongs to.
     *
     * @return \Rubix\ML\ReinforcementLearning\ActionType
     */
    public function actionSpace(): ActionType
    {
        return $this->actionSpace;
    }
}
