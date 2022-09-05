<?php

namespace Rubix\ML\ReinforcementLearning;

/**
 * An environment's observation, to be used by an agent to decide what to do next.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Pablo Duboue
 */
interface Observation
{
    /**
     * Observation space this observation belongs to.
     *
     * @return \Rubix\ML\ReinforcementLearning\ObservationType
     */
    public function observationSpace(): ObservationType;
}
    
