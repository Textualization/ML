<?php

namespace Rubix\ML\ReinforcementLearning\Environments;

use Tensor\Tensor;
use Tensor\Vector;
use Tensor\Matrix;
use Rubix\ML\ReinforcementLearning\Environment;
use Rubix\ML\ReinforcementLearning\Response;
use Rubix\ML\ReinforcementLearning\ActionType;
use Rubix\ML\ReinforcementLearning\ObservationType;
use Rubix\ML\ReinforcementLearning\Action;
use Rubix\ML\ReinforcementLearning\Observation;
use Rubix\ML\ReinforcementLearning\SimpleObservation;
use Rubix\ML\ReinforcementLearning\CompositeObservation;

use function sin;
use function cos;
use function abs;

use const M_PI;


/**
 * A cart pole balancing benchmark. A cart that can be moved left to right with an
 * upright pole on top that needs to be balanced.
 * The environment fails if the angle of the pole exceeds a certain angle.
 * 
 * References:
 * [1] Riedmiller, Peters, Schaal: "Evaluation of Policy Gradient Methods and
 *      Variants on the Cart-Pole Benchmark". ADPRL 2007.
 * [2] cartpole.py in the PyBrain project, by Thomas Rueckstiess
 * [3] Peters J, Vijayakumar S, Schaal S (2003) Reinforcement learning for humanoid robotics
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Pablo Duboue
 */
class CartPole implements Environment
{
    /**
     * Moving left.
     * 
     * @var int
     */
    public const LEFT = 0;
    
    /**
     * Moving right.
     * 
     * @var int
     */
    public const RIGHT = 1;

    /**
     * Time resolution, in seconds
     *
     * @var float
     */
    protected const TAU = 1.0 / 60.0;

    /**
     * Simulation parameter
     * @var float
     */
    protected const MP = 0.1;
    
    /**
     * Simulation parameter
     * @var float
     */
    protected const MC = 1.0;
    
    /**
     * Gravity
     *
     * @var float
     */
    protected const G = 9.81;
    
    /**
     * Force amount for each action
     *
     * @var float
     */
    protected const FORCE_MAG = 10.0;

    /**
     * Delta-T
     *
     * @var float
     */
    protected const DT = 0.02;

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
     * Pole length
     * 
     * @var float
     */
    protected float $polelength;

    /**
     * Angular threshold for failure
     *
     * @var float
     */
    protected float $angularThreshold;

    /**
     * Spatial threshold for failure
     *
     * @var float
     */
    protected float $spatialThreshold;

    /**
     * Sensor state, 4x1 vector of pole angle, pole angle deriv, cart position, cart position deriv
     * @var \Tensor\Vector<float>
     */
    protected Vector $sensors;
    
    /**
     * Number of steps taken so far.
     * @var int
     */
    protected int $steps;
    
    
    /**
     * Create a new cart pole environment.
     */
    public function __construct()
    {
        $this->polelength = 0.5;

        $this->actionSpace = new ActionType(ActionType::DISCRETE, [ 'left', 'right' ]);
        $this->observationSpace = new ObservationType(ObservationType::COMPOSITE, [
            new ObservationType(ObservationType::CONTINUOUS, [ -0.418, 0.418 ]),
            new ObservationType(ObservationType::CONTINUOUS, [ -99999, 99999 ]),
            new ObservationType(ObservationType::CONTINUOUS, [ -4.8, 4.8 ]),
            new ObservationType(ObservationType::CONTINUOUS, [ -99999, 99999 ])
        ]);
        
        $this->angularThreshold = 12 * 2 * M_PI / 360;
        $this->spatialThreshold = 2.4;
        $this->steps = 0;
    }

    /**
     * Transform the sensors into an observation.
     * 
     * @return Observation
     */
    protected function sensorsToObservation() : Observation
    {
        $arr = $this->sensors->asArray();
        $obs = [];
        foreach($arr as $idx => $val) {
            $obs[] = new SimpleObservation($val, $this->observationSpace->params()[$idx]);
        }
        return new CompositeObservation($obs, $this->observationSpace);
    }
    
    /**
     * Indicates the end of an episode.
     *
     * @param ?Vector $sensors new state 
     * @param ?Vector $steps new number of steps
     * @return Response
     */
    public function reset(?Vector $sensors=null, ?int $steps=null) : Response
    {
        $this->steps = $steps ?? 0;
        if($sensors) {
            $this->sensors = $sensors;
        }else{
            $anglePos = Vector::uniform(2)->asArray();
            $this->sensors = Vector::quick([ $anglePos[0] * 0.2, 0.0, $anglePos[1] * 0.5, 0.0 ]);
        }
        return new Response($this->sensorsToObservation(), 0.0, false);
    }

    /**
     * Run a step using the environment dynamics.
     *
     * @param \Rubix\ML\ReinforcementLearning\Action $action
     * @return \Rubix\ML\ReinforcementLearning\Response
     */
    public function step(Action $action): Response
    {
        $this->steps++;
        $force = 0.0;
        switch($action->value()) {
        case CartPole::LEFT:
            $force = -CartPole::FORCE_MAG;  break;
        case CartPole::RIGHT:
            $force = CartPole::FORCE_MAG;  break;
        }
        $this->sensors = $this->rungeKutta4($this->sensors, 0, CartPole::DT, $force);
        $done = abs($this->sensors->offsetGet(0)) > $this->angularThreshold || abs($this->sensors->offsetGet(2)) > $this->spatialThreshold;
        return new Response($this->sensorsToObservation(), 1.0, $done);
    }

    /**
     * Runge-Kutta 4th order over two sample times and using the derivatives function.
     * @param Vector
     * @param float $t0 start time
     * @param float $t1 end time
     * @param float $force force strength
     * @return Vector
     */
    protected function rungeKutta4(Vector $initialState, float $t0, float $t1, float $force) : Vector
    {
        $dt = $t1 - $t0;
        $dt2 = $dt/2.0;
        
        $k1 = $this->derivs($initialState, $t0, $force);
        $k2 = $this->derivs($initialState->add($k1->multiply($dt2)), $t0+$dt2, $force);
        $k3 = $this->derivs($initialState->add($k2->multiply($dt2)), $t0+$dt2, $force);
        $k4 = $this->derivs($initialState->add($k3->multiply($dt)), $t0+$dt, $force);
        return $initialState->add($k1->add($k2->multiply(2))->add($k3->multiply(2))->add($k4)->multiply($dt/6.0));
    }

    /**
     * Derivatives used in the Runge-Kutta.
     * @param Vector $x, point to calculate the derivatives
     * @param float $t, time to calculate the derivatives
     * @param float $force force strength
     * @return Vector
     */
    protected function derivs(Vector $x, float $t, float $force) : Vector
    {
        [ $theta, $thetaDeriv, $s, $sDeriv ]= $x->asArray();
        $u = $thetaDeriv;
        $sin_theta = sin($theta);
        $cos_theta = cos($theta);
        $uDeriv = (CartPole::G * $sin_theta * (CartPole::MC + CartPole::MP) -
               ($force + CartPole::MP * $this->polelength * $theta * $theta * $sin_theta) * $cos_theta) /
            (4.0 / 3.0 * $this->polelength * (CartPole::MC + CartPole::MP) - CartPole::MP * $this->polelength * $cos_theta * $cos_theta);
        $v = $sDeriv;
        $vDeriv = ($force - CartPole::MP * $this->polelength * ($uDeriv * $cos_theta - ($sDeriv * $sDeriv * $sin_theta))) / (CartPole::MC + CartPole::MP);
        return Vector::quick([ $u, $uDeriv, $v, $vDeriv ]);
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
     * Prints the cartpole. Needs the GD extension.
     */
    public function show(?string $path = null) : void
    {
        [ $angle, $angleD, $place, $placeD ] = $this->sensors->asArray();

        $len = 512;
        $middle = $len / 2;
        $img = imagecreatetruecolor($len, $len);

        // colors
        $white = imagecolorallocate($img, 255,255,255);
        $gray = imagecolorallocate($img, 128,128,128);
        $blue = imagecolorallocate($img, 0,0,255);        
        $black = imagecolorallocate($img, 0,0,0);

        // background
        imagefilledrectangle($img, 0,0,$len,$len, $white);
        imagefilledrectangle($img, 0, 505, $len, $len, $gray);
        imagefilledrectangle($img, $middle-2, 0, $middle+2, $len, $gray);        

        $factor = $middle / 2.4;
        $pos = $place * $factor + $middle;
        
        // cart
        imagefilledrectangle($img, $pos-50, 400, $pos+50, 500, $blue);
        imagefilledellipse($img, $pos-25, 500, 10,10, $black);
        imagefilledellipse($img, $pos+25, 500, 10,10, $black);
        // pole
        imageline($img, $pos, 400,  $pos + 200 * sin($angle), 400 - 200 * cos($angle), $black);

        // texts
        imagestring($img, 4, 10, 10, "Position:    {$this->sensors[2]}", $black);
        imagestring($img, 4, 10, 30, "P. Velocity: {$this->sensors[3]}", $black);
        imagestring($img, 4, 10, 50, "Angle:       {$this->sensors[0]}", $black);
        imagestring($img, 4, 10, 70, "A. Velocity: {$this->sensors[1]}", $black);
        imagestring($img, 4, 10, 90, "Steps:       {$this->steps}", $black);

        imagepng($img, $path);
    }

    /**
     * Number of steps taken so far.
     */
    public function steps() : int
    {
        return $this->steps;
    }
}
    
