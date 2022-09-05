<?php

namespace Rubix\ML\ReinforcementLearning;

/**
 * The type of an action.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Pablo Duboue
 */
class ActionType
{
    /**
     * A discrete action. Params are the possible values.
     * 
     * @var int
     */
    public const DISCRETE = 1;

    /**
     * A continuous action. Params are maximum and minimum.
     * 
     * @var int
     */
    public const CONTINUOUS = 1;

    /**
     * An action with subactions. Params are the ActionType of the sub-actions.
     *
     * @var int
     */
    public const COMPOSITE = 2;

    /**
     * The action type
     * 
     * @var int
     */
    protected int $type;

    /**
     * Its parameters
     *
     * @var list<string>|list<float>|list<ActionType>
     */
    protected mixed $params;

    /**
     * Create an action type
     *
     * @param int $type
     * @param list<string>|list<float>|list<ActionType> $params
     */
    public function __construct(int $type, mixed $params)
    {
        $this->type = $type;
        //TODO: verify types of params
        $this->params = $params;
    }

    /**
     * The type of the action
     *
     * @return int
     */
    public function type() : int
    {
        return $this->type;
    }

    /**
     * The parameters of the action type
     *
     * @return list<string>|list<float>|list<ActionType>
     */
    public function params() : mixed
    {
        return $this->params;
    }

    /**
     * Verify that an action belongs in this action space.
     *
     * @param Action $action
     * @return bool
     */
    public function contains(Action $action) : bool
    {
        //TODO
        return true;
    }
} 
