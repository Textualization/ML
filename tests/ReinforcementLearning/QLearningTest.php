<?php

namespace Rubix\ML\Tests\ReinforcementLearning;

use Rubix\ML\ReinforcementLearning\QLearning;
use Rubix\ML\ReinforcementLearning\Environments\Maze;
use PHPUnit\Framework\TestCase;

/**
 * @group ReinforcementLearning
 * @covers \Rubix\ML\ReinforcementLearning\QLearning
 */
class QLearningTest extends TestCase
{
    /**
     * The maximum number of steps until giving up.
     *
     * @var int
     */
    protected const MAX_STEPS = 100;

    /**
     * The environment over which to run the test.
     * @var Maze
     */
    protected Maze $env;

    
    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->env = new Maze(10, 10, [0, 0], [9, 9], [ [1,1],[1,2],[1,3],[1,4],[1,6],[1,7],[1,8],[1,9],
                                                        [2,1],[2,2],
                                                        [3,2],[3,3],[3,4],[3,5],[3,6],[3,8],[3,9],
                                                        [4,2],[4,8],[4,9],
                                                        [5,1],[5,2],[5,3],[5,4],[5,5],[5,8],[5,9],
                                                        [6,0],[6,1],[6,2],[6,3],[6,4],[6,7],[6,8],
                                                        [7,4],[7,5],[7,7],
                                                        [8,1],[8,2],[8,5],[8,7],[8,8],[8,9] ], 1000.0);
    }

    protected function stepsToSuccess(QLearning $qlearning) : float
    {
        $average = 0.0;
        for($i=0; $i<10; $i++) {
            $response = $this->env->reset();
            $steps = 0;
            while(! $response->finished() && $steps < QLearningTest::MAX_STEPS) {
                $action = $qlearning->execute($response->observation());
                $response = $this->env->step($action);
                $steps++;
            }
            $average = $steps / 10.0;
        }
        return $average;
    }
    
    /**
     * @test
     */
    public function build() : void
    {
        // base system
        $qlearning = new QLearning($this->env, 0.8, 0.9);
        $steps_at_0 = $this->stepsToSuccess($qlearning);

        // train 1
        for($i=0;$i<100;$i++) {
            $qlearning->trainEpisode($this->env, 0.05, 0.002);
        }
        $steps_at_1 = $this->stepsToSuccess($qlearning);
        $this->assertLessThan($steps_at_0, $steps_at_1);
    }
}
