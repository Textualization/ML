<?php

namespace Rubix\ML\ReinforcementLearning;

/**
 * An environment's observation that encompasses multiple observations.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Pablo Duboue
 */
class CompositeObservation implements Observation
{
    /**
     * The values of this observation
     *
     * @var list<SimpleObservation>
     */
    protected array $values;

    /**
     * The type of this observation
     *
     * @var ObservationType
     */
    protected ObservationType $type;

    /**
     * Create an observation.
     */
    public function __construct(array $values, ObservationType $type)
    {
        $this->values = $values;
        //TODO validate type to be composite and count(values) equal to count($type->params())
        $this->type = $type;
    }

    /**
     * The value of this observation.
     *
     * @return list<float|int>
     */
    public function value() : array
    {
        $result = [];
        foreach($this->values as $value) {
            $result[] = $value->value();
        }
        return $result;
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
    
