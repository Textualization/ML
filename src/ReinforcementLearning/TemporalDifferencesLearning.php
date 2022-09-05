<?php

namespace Rubix\ML\ReinforcementLearning;

/**
 * An abstract class to base Q-Learning subclasses.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Pablo Duboue
 */
abstract class TemporalDifferencesLearning
{

    /**
     * Possible actions.
     *
     * @var ActionType
     */
    protected ActionType $actionSpace;

    /**
     * Possible states.
     *
     * @var ObservationType
     */
    protected ObservationType $observationSpace;
        
    /**
     * Gamma parameter that controls the trade-off of future rewards versus immediate rewards. A larger value means the future is more important.
     *
     * @var float
     */
    protected float $gamma;
    
    /**
     * Learning rate, it controls the balance between exploration and exploitation.
     *
     * @var float
     */
    protected float $epsilon;
    
    /**
     * Create a Temporal Differences instance for a given Environment.
     *
     * @param Environment
     * @param float $gamma controls whether immediate or future rewards are more important (zero means only immediate rewards matter)
     * @param float $epsilon controls whether suboptimal actions will be taken to explore the space (zero means only optimal actions taken, leads to local optimum)
     */
    public function __construct(Environment $env, float $gamma, float $epsilon)
    {
        $this->actionSpace = $env->actionSpace();
        $this->observationSpace = $env->observationSpace();
        //TODO validate gamma \in [0, 1]
        $this->gamma = $gamma;
        //TODO validate epsilon \in [0, 1]
        $this->epsilon = $epsilon;
    }

    /**
     * Update an expected value for a given observation/action pair.
     * It includes a learning rate.
     * @param Observation
     * @param Action
     * @param float $learningRate
     * @param float $value
     */
    abstract protected function updateValue(Observation $observation, Action $action, float $learningRate, float $value) : void;

    /**
     * Find the maximum possible value for a given observation, over all possible actions.
     * 
     * @param Observation
     * @return float the value
     */
    abstract protected function maxValueAction(Observation $observation) : float;

    /**
     * Given a state, return the action to take next. 
     * It uses $epsilon to balance exploration vs. exploitation. 
     * An $epsilon of 0 picks the action with highest value.
     *
     * @param Observation $observation
     * @param float
     * @return Action
     */
    abstract protected function explorationPolicy(Observation $observation, float $epsilon) : Action;

    /**
     * Train one episode against a given environment.
     * It includes a learning rate with decay.
     *
     * @param Environmnet $env
     * @param float $learningRate
     * @param float $decay
     */
    public function trainEpisode(Environment $env, float $learningRate, float $decay) : void
    {
        $response = $env->reset();
        $current = $response->observation();
        $steps = 0;
        while(! $response->finished()) {
            $nextAction = $this->explorationPolicy($current, $this->epsilon);
            $response = $env->step($nextAction);
            $maxValue = $this->maxValueAction($response->observation());
            $lr = $learningRate / (1 + $steps * $decay);
            $newValue = $response->reward() + $this->gamma * $maxValue;
            $this->updateValue($current, $nextAction, $lr, $newValue);
            $current = $response->observation();
            $steps++;
        }
    }

    /**
     * Use learned values to pick the next action
     *
     * @param Observation
     * @return Action
     */
    public function execute(Observation $observation) : Action
    {
        return $this->explorationPolicy($observation, 0);
    }

    /**
     * Get the epsilon parameter that controls exploration vs exploitation trade-offs.
     * @return float
     */
    public function epsilon() : float
    {
        return $this->epsilon;
    }
    
    /**
     * Set the epsilon parameter that controls exploration vs exploitation trade-offs.
     * @param float
     */
    public function updateEpsilon(float $epsilon) : void
    {
        //TODO validate
        $this->epsilon = $epsilon;
    }
}
