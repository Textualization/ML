<?php

namespace Rubix\ML\ReinforcementLearning;

/**
 * Reinforcement Learning environment response.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Pablo Duboue
 */
class Response
{
    
    /**
     * The observation at this step
     *
     * @var \Rubix\ML\ReinforcementLearning\Observation
     */
    protected \Rubix\ML\ReinforcementLearning\Observation $observation;

    /**
     * The reward at this step
     *
     * @var float
     */
    protected float $reward;

    /**
     * Whether we are finished.
     *
     * @var bool
     */
    protected bool $finished;

    /**
     * @param \Rubix\ML\ReinforcementLearning\Observation $observation
     * @param float $reward
     * @param bool $finished
     */
    public function __construct(Observation $observation, float $reward, bool $finished)
    {
        $this->observation = $observation;
        $this->reward = $reward;
        $this->finished = $finished;
    }

    /**
     * Return the observation.
     * 
     * @return \Rubix\ML\ReinforcementLearning\Observation
     */
    public function observation() : Observation
    {
        return $this->observation;
    }
    
    /**
     * Return the reward.
     * 
     * @return float
     */
    public function reward() : float
    {
        return $this->reward;
    }
    
    /**
     * Return whether we are finished.
     * 
     * @return \Rubix\ML\ReinforcementLearning\Observation
     */
    public function finished() : float
    {
        return $this->finished;
    }
}
