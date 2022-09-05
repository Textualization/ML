<?php

namespace Rubix\ML\ReinforcementLearning;

/**
 * An action among a list of discrete options.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Pablo Duboue
 */
class DiscreteAction implements Action
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
     * @var int
     */
    protected int $value;

    /**
     * Create a discrete action.
     *
     * @param int $value
     * @param ActionType $actionSpace
     */
    public function __construct(int $value, ActionType $actionSpace)
    {
        $this->value = $value;
        //TODO validate $actionSpace is Discrete
        //TODO validate $value is within range
        $this->actionSpace = $actionSpace;
    }

    /**
     * The selected value.
     *
     * @return int
     */
    public function value() : int
    {
        return $this->value;
    }

    /**
     * The selected value as a string.
     *
     * @return string
     */
    public function valueString() : string
    {
        return $this->actionSpace->params()($this->value);
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
