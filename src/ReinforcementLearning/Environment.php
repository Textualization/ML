<?php

namespace Rubix\ML\ReinforcementLearning;

/**
 * Reinforcement Learning Environment
 *
 * The environment in which a reinforcement learning agent interacts.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Pablo Duboue
 */
interface Environment
{
    /**
     * Indicate the end of an episode.
     *
     * @return \Rubix\ML\ReinforcementLearning\Response
     */
    public function reset() : Response;

    /**
     * Run a step using the environment dynamics.
     *
     * @param \Rubix\ML\ReinforcementLearning\Action $action
     * @return \Rubix\ML\ReinforcementLearning\Response
     */
    public function step(Action $action): Response;

    /**
     * (Optional) List possible actions.
     *
     * @return list<\Rubix\ML\ReinforcementLearning\ActionType>
     */
    public function actionSpace(): ActionType;
    
    /**
     * (Optional) List possible observations.
     *
     * @return list<\Rubix\ML\ReinforcementLearning\ObservationType>
     */
    public function observationSpace(): ObservationType;

    /**
     * Print the environment to the screen or to a file.
     * 
     * @param ?file $path
     */
    public function show(?string $path = null) : void;
}
