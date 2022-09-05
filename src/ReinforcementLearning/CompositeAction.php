<?php

namespace Rubix\ML\ReinforcementLearning;

/**
 * An action composed by sub-actions.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Pablo Duboue
 */
class CompositeAction implements Action
{
    /**
     * The action space
     *
     * @var ActionType
     */
    protected ActionType $actionSpace;

    /**
     * The subaction values.
     *
     * @var list<Action>
     */
    protected array $values;

    /**
     * Create a continuous action.
     *
     * @param list<Action> $values
     * @param ActionType $actionSpace
     */
    public function __construct(array $values, ActionType $actionSpace)
    {
        $this->values = $values;
        //TODO validate $actionSpace is Composite
        //TODO validate $values are all either Discrete or Continuous
        $this->actionSpace = $actionSpace;
    }

    /**
     * The selected value.
     *
     * @return list<float|int>
     */
    public function value() : array
    {
        $result = [];
        foreach($values as $act) {
            $result[] = $act->value();
        }
        return $result;
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
