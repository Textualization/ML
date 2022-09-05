<?php

namespace Rubix\ML\ReinforcementLearning\Environments;

use Rubix\ML\ReinforcementLearning\Environment;
use Rubix\ML\ReinforcementLearning\Response;
use Rubix\ML\ReinforcementLearning\ActionType;
use Rubix\ML\ReinforcementLearning\ObservationType;
use Rubix\ML\ReinforcementLearning\Action;
use Rubix\ML\ReinforcementLearning\Observation;
use Rubix\ML\ReinforcementLearning\SimpleObservation;
use Rubix\ML\ReinforcementLearning\CompositeObservation;

/**
 * Sample environment with a 2D-maze
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Pablo Duboue
 */
class Maze implements Environment
{
    /**
     * Moving up.
     * 
     * @var int
     */
    public const UP = 0;
    
    /**
     * Moving down.
     * 
     * @var int
     */
    public const DOWN = 1;
    
    /**
     * Moving left.
     * 
     * @var int
     */
    public const LEFT = 2;
    
    /**
     * Moving right.
     * 
     * @var int
     */
    public const RIGHT = 3;
    
    /**
     * The maze as a 2D array, a value of true means there is a wall there.
     *
     * @var list<list<boolean>>
     */
    protected array $maze;

    /**
     * The width of the maze
     *
     * @var int
     */
    protected int $width;
    
    /**
     * The height of the maze
     *
     * @var int
     */
    protected int $height;
    

    /**
     * x, y positions of the stating point
     *
     * @var list<int>
     */
    protected array $start;
    
    /**
     * x, y positions of the current position of the agent.
     *
     * @var list<int>
     */
    protected array $current;
    
    /**
     * x, y positions of the goal point.
     *
     * @var list<int>
     */
    protected array $goal;

    /**
     * Exit reward, the only non-zero reward offered.
     *
     * @var float
     */
    protected float $exitReward;
    
    /**
     * The action space
     *
     * @var ActionType
     */
    protected ActionType $actionSpace;
    
    /**
     * The observation space
     *
     * @var ObservationType
     */
    protected ObservationType $observationSpace;
    
    /**
     * Create a new maze of a given size with given walls and start and end points.
     *
     * @param int $width
     * @param int $height
     * @param list<int> $start
     * @param list<int> $goal
     * @param list<list<int>> $walls
     * @param float $exitReward
     */
    public function __construct(int $width, int $height, array $start, array $goal, array $walls, float $exitReward)
    {
        //TODO validate $width > 0
        //TODO validate $height > 0
        $this->width = $width;
        $this->height = $height;
        
        $this->maze = [];
        for($i=0; $i<$height; $i++) {
            $row = [];
            for($j=0; $j<$width; $j++) {
                $row[] = false;
            }
            $this->maze[] = $row;
        }
        foreach($walls as $point){
            //TODO validate point within [0,$width-1] [0,$height-1]
            $this->maze[$point[0]][$point[1]] = true;
        }
        //TODO validate $start, $goal are not walls
        //TODO validate $exitReward > 0
        $this->start = $start;
        $this->goal = $goal;
        $this->exitReward = $exitReward;
        $this->current = $start;

        $this->actionSpace = new ActionType(ActionType::DISCRETE, [ 'up', 'down', 'left', 'right' ]);
        
        $xvals = [];
        for($i=0; $i<$height; $i++) {
            $xvals[] = $i;
        }
        $yvals = [];
        for($i=0; $i<$width; $i++) {
            $yvals[] = $i;
        }
        $this->observationSpace = new ObservationType(ObservationType::COMPOSITE, [
            new ObservationType(ObservationType::DISCRETE, $xvals),
            new ObservationType(ObservationType::DISCRETE, $yvals)
        ]);
    }
    
    /**
     * Indicate the end of an episode.
     *
     * @return Response
     */
    public function reset() : Response
    {
        $this->current = $this->start;
        return new Response(new CompositeObservation([ new SimpleObservation($this->current[0], $this->observationSpace->params()[0]),
                                                       new SimpleObservation($this->current[1], $this->observationSpace->params()[1]) ],
                                                     $this->observationSpace),
                            0.0, false);
    }

    /**
     * Run a step using the environment dynamics.
     *
     * @param \Rubix\ML\ReinforcementLearning\Action $action
     * @return \Rubix\ML\ReinforcementLearning\Response
     */
    public function step(Action $action): Response
    {
        switch($action->value()) {
        case Maze::UP:
            if($this->current[0] > 0 && !$this->maze[$this->current[0]-1][$this->current[1]]) {
                $this->current[0] -= 1;
            }
            break;
        case Maze::DOWN:
            if($this->current[0] < $this->height - 1 && !$this->maze[$this->current[0]+1][$this->current[1]]) {
                $this->current[0] += 1;
            }
            break;
        case Maze::LEFT:
            if($this->current[1] > 0 && !$this->maze[$this->current[0]][$this->current[1]-1]) {
                $this->current[1] -= 1;
            }
            break;
        case Maze::RIGHT:
            if($this->current[1] < $this->width - 1 && !$this->maze[$this->current[0]][$this->current[1]+1]) {
                $this->current[1] += 1;
            }
            break;
        }
        $found = $this->current[0] == $this->goal[0] && $this->current[1] == $this->goal[1];
        return new Response(new CompositeObservation([ new SimpleObservation($this->current[0], $this->observationSpace->params()[0]),
                                                       new SimpleObservation($this->current[1], $this->observationSpace->params()[1]) ],
                                                     $this->observationSpace),
                            $found ? $this->exitReward: 0.0, $found);
    }

    /**
     * (Optional) List possible actions.
     *
     * @return list<\Rubix\ML\ReinforcementLearning\ActionType>
     */
    public function actionSpace(): ActionType
    {
        return $this->actionSpace;
    }
    
    /**
     * (Optional) List possible observations.
     *
     * @return list<\Rubix\ML\ReinforcementLearning\ObservationType>
     */
    public function observationSpace(): ObservationType
    {
        return $this->observationSpace;
    }

    /**
     * Print maze to screen.
     */
    public function show() : void
    {
        for($i=0; $i<$this->height; $i++) {
            for($j=0; $j<$this->width; $j++) {
                if($this->current[0] == $i && $this->current[1] == $j){
                    echo "o";
                }elseif($this->goal[0] == $i && $this->goal[1] == $j){
                    echo "X";
                }elseif($this->maze[$i][$j]){
                    echo "#";
                }else{
                    echo ".";
                }
            }
            echo "\n";
        }
    }
}
    
