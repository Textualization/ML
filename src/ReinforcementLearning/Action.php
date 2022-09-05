<?php

namespace Rubix\ML\ReinforcementLearning;

/**
 * Reinforcement Learning action, as executed by the agent.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Pablo Duboue
 */
interface Action
{
    /**
     * The action value.
     *
     * @return mixed
     */
    public function value();

    /**
     * Action space this action belongs to.
     *
     * @return \Rubix\ML\ReinforcementLearning\ActionType
     */
    public function actionSpace(): ActionType;
}
    
