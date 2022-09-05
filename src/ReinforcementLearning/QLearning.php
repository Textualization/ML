<?php

namespace Rubix\ML\ReinforcementLearning;

/**
 * The Q-Learning algorithm.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Pablo Duboue
 */
class QLearning
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
     * The Q-table
     *
     * @var list<list<float>>
     */
    protected array $qtable;
    
    /**
     * Create a Q-Learning instance for a given Environment.
     */
    public function __construct(Environment $env, float $gamma)
    {
        //TODO validate action space is discrete
        $this->actionSpace = $env->actionSpace();
        $this->numActions = count($this->actionSpace->params());
        //TODO validate observation space is discrete
        $this->observationSpace = $env->observationSpace();
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
        $this->gamma = $gamma;
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
     * Train one episode against a given environment.
     *
     * @param Environmnet $env
     */
    public function trainEpisode(Environment $env) : void
    {
        $response = $env->reset();
        $current = $this->observationToState($response->observation());
        while(! $response->finished()) {
            $nextAction = mt_rand(0, $this->numActions);
            $response = $env->step(new DiscreteAction($nextAction, $this->actionSpace));
            $nextState = $this->observationToState($response->observation());
            $maxQ = -999999;
            for($i=0;$i<$this->numActions;$i++){
                $maxQ = max($maxQ, $this->qtable[$nextState][$i]);
            }
            $this->qtable[$current][$nextAction] = $response->reward() + $this->gamma * $maxQ;
            $current = $nextState;
        }
    }

    /**
     * Use the qtable to pick the next action
     *
     * @param Observation
     * @return Action
     */
    public function execute(Observation $observation) : Action
    {
        $state = $this->observationToState($observation);
        $bestAction = -1;
        $bestQ = -1;
        for($i=0;$i<$this->numActions;$i++){
            if($bestAction == -1 || $bestQ < $this->qtable[$state][$i]) {
                $bestAction = $i;
                $bestQ = $this->qtable[$state][$i];
            }
        }
        return new DiscreteAction($bestAction, $this->actionSpace);
    }
}
