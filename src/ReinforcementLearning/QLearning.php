<?php

namespace Rubix\ML\ReinforcementLearning;

/**
 * The Q-Learning algorithm.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Pablo Duboue
 */
class QLearning extends TemporalDifferencesLearning
{

    /**
     * Number of states. Derived from the environment.
     *
     * @var int
     */
    protected int $numStates;

    /**
     * Number of actions. Derived from the environment.
     *
     * @var int
     */
    protected int $numActions;
    
    /**
     * The Q-table
     *
     * @var list<list<float>>
     */
    protected array $qtable;
    
    /**
     * Create a Q-Learning instance for a given Environment. The environments has to have discrete actions and observations.
     * 
     * @param Environment
     * @param float $gamma controls whether immediate or future rewards are more important (zero means only immediate rewards matter)
     * @param float $epsilon controls whether suboptimal actions will be taken to explore the space (zero means only optimal actions taken, leads to local optimum)
     */
    public function __construct(Environment $env, float $gamma, float $epsilon)
    {
        parent::__construct($env, $gamma, $epsilon);
        //TODO validate action space is discrete
        $this->numActions = count($this->actionSpace->params());
        //TODO validate observation space is discrete
        if($this->observationSpace->type() == ObservationType::COMPOSITE) {
            $this->numStates = 1;
            foreach($this->observationSpace->params() as $obs){
                $this->numStates *= count($obs->params());
            }
        }else{
            $this->numStates = count($this->observationSpace->params());
        }
        $this->qtable = [];
        for($x=0;$x<$this->numStates;$x++){
            $row=[];
            for($y=0;$y<$this->numActions;$y++){
                $row[]=0.0;
            }
            $this->qtable[] = $row;
        }
    }

    /**
     * Transform a potentially composite observation into a state number.
     *
     * @param Observation
     * @return int
     */
    protected function observationToState(Observation $obs): int
    {
        if($obs->observationSpace()->type() == ObservationType::COMPOSITE) {
            $cols = $obs->observationSpace()->params();
            $dims = [];
            foreach($cols as $obsType) {
                $dims[] = count($obsType->params());
            }
            $result = 0;
            foreach($obs->value() as $coord => $val) {
                $result *= $dims[$coord];
                $result += $val;
            }
            return $result;
        }else{
            return $obs->value();
        }
    }

    /**
     * Update an expected value for a given observation/action pair.
     * It includes a learning rate.
     * @param Observation
     * @param Action
     * @param float $learningRate
     * @param float $value
     */
    protected function updateValue(Observation $observation, Action $action, float $learningRate, float $value) : void
    {
        $state = $this->observationToState($observation);
        $actNum = $action->value();
        $this->qtable[$state][$actNum] *= 1.0 - $learningRate;
        $this->qtable[$state][$actNum] = $learningRate * $value;
    }

    /**
     * Find the maximum possible value for a given observation, over all possible actions.
     * 
     * @param Observation
     * @return float the value
     */
    protected function maxValueAction(Observation $observation) : float
    {
        $state = $this->observationToState($observation);
        $bestQ = -1;
        for($i=0; $i<$this->numActions; $i++) {
            if($bestQ == -1 || $bestQ < $this->qtable[$state][$i]) {
                $bestQ = $this->qtable[$state][$i];
            }
        }
        return $bestQ;
    }

    /**
     * Given a state, return the action to take next. 
     * It uses $epsilon to balance exploration vs. exploitation. 
     * An $epsilon of 0 picks the action with highest value.
     *
     * @param Observation $observation
     * @param float
     * @return Action
     */
    protected function explorationPolicy(Observation $observation, float $epsilon) : Action
    {
        $action = -1;
        if($epsilon > 0 && mt_rand() / mt_getrandmax() < $epsilon) {
            $action = mt_rand(0, $this->numActions - 1);
        }else{
            $state = $this->observationToState($observation);
            $bestQ = -1;
            for($i=0; $i<$this->numActions; $i++) {
                if($action == -1 || $bestQ < $this->qtable[$state][$i]) {
                    $action = $i;
                    $bestQ = $this->qtable[$state][$i];
                }
            }
        }
        return new DiscreteAction($action, $this->actionSpace);
    }
}
