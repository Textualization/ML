<?php

namespace Rubix\ML\ReinforcementLearning;

/**
 * An environment's observation, to be used by an agent to decide what to do next.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Pablo Duboue
 */
class SimpleObservation implements Observation
{
    /**
     * The value of this observation
     *
     * @var float|int
     */
    protected mixed $value;

    /**
     * The type of this observation
     *
     * @var ObservationType
     */
    protected ObservationType $type;

    /**
     * Create an observation.
     */
    public function __construct(mixed $value, ObservationType $type)
    {
        $this->value = $value;
        //TODO validate type is either continuous or discrete
        $this->type = $type;
    }

    /**
     * The value of this observation.
     *
     * @return float|int
     */
    public function value() : mixed
    {
        return $this->value;
    }
    
    /**
     * Observation space this observation belongs to.
     *
     * @return \Rubix\ML\ReinforcementLearning\ObservationType
     */
    public function observationSpace(): ObservationType
    {
        return $this->type;
    }
}
    
