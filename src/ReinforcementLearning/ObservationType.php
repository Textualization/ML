<?php

namespace Rubix\ML\ReinforcementLearning;

/**
 * The type of observations.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Pablo Duboue
 */
class ObservationType
{
    /**
     * An observation with a single, discrete value. Params are the possible values.
     * 
     * @var int
     */
    public const DISCRETE = 1;

    /**
     * An observation with a single, continuous value. Params are maximum and minimum.
     * 
     * @var int
     */
    public const CONTINUOUS = 2;

    /**
     * An observation with multiple values. Params are the sub-observation types, all of them either DISCRETE or CONTINUOUS.
     *
     * @var int
     */
    public const COMPOSITE = 2;

    /**
     * The observation type
     * 
     * @var int
     */
    protected int $type;

    /**
     * Its parameters
     *
     * @var list<string>|list<float>|list<ObservationType>
     */
    protected mixed $params;
    
    /**
     * Create an observation type
     *
     * @param int $type
     * @param list<string>|list<float>|list<ObservationType> $params
     */
    public function __construct(int $type, mixed $params) {
        $this->type = $type;
        //TODO: verify types of params
        $this->params = $params;
    }

    /**
     * The type of the observation
     *
     * @return int
     */
    public function type() : int
    {
        return $this->type;
    }

    /**
     * The parameters of the observation type
     *
     * @return list<string>|list<float>|list<ObservationType>
     */
    public function params() : mixed
    {
        return $this->params;
    }
} 
