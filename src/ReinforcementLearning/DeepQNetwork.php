<?php

namespace Rubix\ML\ReinforcementLearning;

use Tensor\Matrix;
use Rubix\ML\NeuralNet\Network;
use Rubix\ML\NeuralNet\FeedForward;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Input;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\MultiContinuous;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\ReLU;
use Rubix\ML\NeuralNet\CostFunctions\HuberLoss;
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;
use Rubix\ML\NeuralNet\Layers\Placeholder1D;
use Rubix\ML\NeuralNet\Initializers\Xavier2;
use Rubix\ML\NeuralNet\Initializers\Constant;


/**
 * The Deep QNetwork using actor-critic.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Pablo Duboue
 */
class DeepQNetwork extends TemporalDifferencesLearning
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
     * @var FeedForward
     */
    protected FeedForward $actor;
    
    /**
     * Critic network, from input neurons to numActions continuous.
     *
     * @var FeedForward
     */
    protected FeedForward $critic;

    /**
     * Replay memory, triples <state (input layer), action (int), target value (float)>
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
        $output = new MultiContinuous(new HuberLoss());
        $optimizer = $optimizer ?? new Adam(0.001);
        $layers = $layers ?? [ new Dense(($this->numInputs + $this->numActions) / 2),
                               new Activation(new ReLU()) ];
        $layers[] = new Dense($this->numActions, 0.0, true, new Xavier2(), new Constant(0.1));
        $this->actor = new Feedforward($input, $layers, $output, $optimizer);
        $this->actor->initialize();
        $this->critic = clone $this->actor;
        $this->replayMemory = [];
        $this->replayMemoryMaxSize = 2000;
        $this->replayMemoryLastItem = 0;
    }

    /**
     * Set the replay memory maximum size
     * @param int
     */
    public function replayMemoryMaxSize(int $maxSize) : void
    {
        $this->replayMemoryMaxSize = $maxSize;
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
                    $result[$idx + $val] = 1.0;
                    $idx += count($cols[$coord]->params());
                }else{
                    $result[$idx] = $val;
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
        if($this->replayMemoryLastItem > $this->replayMemoryMaxSize) {
            unset($this->replayMemory[$this->replayMemoryLastItem - $this->replayMemoryMaxSize]);
        }
        $this->replayMemory[$this->replayMemoryLastItem] = [ $activation, $action->value(), $value];
    }

    /**
     * Find the maximum possible value for a given observation, over all possible actions.
     * 
     * @param Observation
     * @return float the value
     */
    protected function maxValueAction(Observation $observation) : float
    {
        $activation = $this->observationToActivation($observation);
        $input = Matrix::quick([ $activation ])->transpose();
        return $this->actor->feed($input)->max()->max();
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
            $activation = $this->observationToActivation($observation);
            $input = Matrix::quick([ $activation ])->transpose();
            $output = $this->actor->feed($input)->asArray();
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

    /**
     * Train a minibatch, returns the loss.
     *
     * @param int $batchSize how many entries to sample from replay memory.
     * @return float the loss for the minibatch
     */
    public function trainBatch(int $batchSize) : float
    {
        shuffle($this->replayMemory);
        $inputs=[];
        $outputs=[];
        $count = 0;
        foreach($this->replayMemory as $entry){
            $outputs[] = [$entry[1], $entry[2]];
            $inputs[] = $entry[0];
            $count++;
            if($count >= $batchSize){
                break;
            }
        }
        $input = Matrix::quick($inputs)->transpose();
        $computed = $this->critic->feed($input)->transpose()->asArray();
        $output = $computed;
        foreach($outputs as $idx => $e) {
            $output[$idx][$e[0]] = $e[1];
        }
        return $this->critic->backpropagate($output);
    }

    /**
     * Adopt the trained critic as a new actor.
     */
    public function adoptCritic() : void {
        $this->actor = clone $this->critic;
        $this->replayMemory = [];
        $this->replayMemoryLastItem = 0;
    }
}
