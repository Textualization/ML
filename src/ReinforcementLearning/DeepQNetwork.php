<?php

namespace Rubix\ML\ReinforcementLearning;

use Rubix\ML\NeuralNet\Network;
use Rubix\ML\NeuralNet\FeedForward;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Input;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\ReLU;


/**
 * The Deep Q-Learning algorithm using actor-critic.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Pablo Duboue
 */
class DeepQLearning extends TemporalDifferencesLearning
{

    /**
     * Number of actions. Derived from the environment.
     *
     * @var int
     */
    protected int $numActions;
    
    /**
     * Number of input neurons. Derived from the environment.
     *
     * @var int
     */
    protected int $numInputs;
    
    /**
     * Actor network, from input neurons to numActions continuous.
     *
     * @var 
     */
    protected array Network $actor;
    
    /**
     * Critic network, from input neurons to numActions continuous.
     *
     * @var list<list<float>>
     */
    protected Network $critic;

    /**
     * Replay memory, 5-tuples <state, action, target>
     * 
     * @var list<list<float>>
     */
    protected array $replayMemory;

    /**
     * Replay memory max size.
     *
     * @var int
     */
    protected int $replayMemoryMaxSize;

    /**
     * Replay memory last item.
     *
     * @var int
     */
    protected int $replayMemoryLastItem;
    
    /**
     * Create a Deep Q-Learning instance for a given Environment. The environments has to have discrete actions.
     * 
     * @param Environment
     * @param float $gamma controls whether immediate or future rewards are more important (zero means only immediate rewards matter)
     * @param float $epsilon controls whether suboptimal actions will be taken to explore the space (zero means only optimal actions taken, leads to local optimum)
     * @param ?list<Layer> the hidden layers of the DQN (optional)
     * @param ?Optimizer the optimizer to use (optional)
     */
    public function __construct(Environment $env, float $gamma, float $epsilon,
                                ?array $layers = null, ?Optimizer $optimizer = null)
    {
        parent::__construct($env, $gamma, $epsilon);
        //TODO validate action space is discrete
        $this->numActions = count($this->actionSpace->params());
        if($this->observationSpace->type() == ObservationType::COMPOSITE) {
            $this->numInputs = 0;
            foreach($this->observationSpace->params() as $obs){
                if($obs->type() == ObservationType::CONTINUOUS) {
                    $this->numInputs += 1;
                }elseif($obs->type() == ObservationType::DISCRETE) {
                    $this->numInputs += count($obs->params()); // one-hot encoded
                }
            }
        }elseif($this->observationSpace->type() == ObservationType::CONTINUOUS) {
            $this->numInputs = 1;
        }elseif($this->observationSpace->type() == ObservationType::DISCRETE) {
            $this->numInputs = count($this->observationSpace->params());  // one-hot encoded
        }
        $input = new Placeholder1D($this->numInputs);
        $output = new MultiContinuous(new HuberLoss()),
        $optimizer = $optimizer ?? new Adam();
        $layers = $layers ?? [ new Dense(($numInputs + $numActions) / 2),
                               new Activation(new ReLU()),
                               new Dense($numActions) ];
        $this->actor = new Feedforward($input, $layers, $output, $optimizer);
        $this->network->initialize();
        $this->critic = clone $this->actor;
        $this->replayMemory = [];
        $this->replayMemoryMaxSize = 2000;
        $this->replayMemoryLastItem = 0;
    }

    /**
     * Transform a potentially composite observation into a state number.
     *
     * @param Observation
     * @return list<float>
     */
    protected function observationToActivation(Observation $obs): array
    {
        $result=[];
        for($i=0; $i<$this->numInputs; $i++) {
            $result[] = 0.0;
        }
        if($obs->observationSpace()->type() == ObservationType::COMPOSITE) {
            $idx = 0;
            $cols = $obs->observationSpace()->params();
            foreach($obs->value() as $coord => $val) {
                if($cols[$coord]->type() == ObservationType::DISCRETE) {
                    $result[$idx + $obs->value()] = 1.0;
                    $idx += count($cols[$coord]->params());
                }else{
                    $result[$idx] = $obs->value();
                    $idx++;
                }
            }
        }elseif($obs->type() == ObservationType::DISCRETE) {
            $result[$obs->value()] = 1.0;
        }else{
            $result[0] = $obs->value();
        }
        return $result;
    }

    /**
     * Stores in replay memory.
     *
     * @param Observation
     * @param Action
     * @param float $learningRate ignored
     * @param float $value
     */
    protected function updateValue(Observation $observation, Action $action, float $learningRate, float $value) : void
    {
        $activation = $this->observationToActivation($observation);
        $this->replayMemoryLastItem++;
        if($this->replayMemoryLastItem > $this->replayMemorySize) {
            unset($replayMemory[$replayMemoryLastItem - $this->replayMemorySize]);
        }
        $entry = $activation;
        $entry[] = $action->value();
        $entry[] = $value;
        $this->replayMemory[$this->replayMemoryLastItem] = $entry;
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
            $input = $this->observationToState($observation);
            $output = $this->actor->infer($input);
            $bestQ = -1;
            for($i=0; $i<$this->numActions; $i++) {
                if($action == -1 || $bestQ < $output[$i]) {
                    $action = $i;
                    $bestQ = $output[$i];
                }
            }
        }
        return new DiscreteAction($action, $this->actionSpace);
    }
}
