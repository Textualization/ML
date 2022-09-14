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

use function array_slice;
use function sort;
use function sum;


/**
 * A time-series trading environment. The agent has a certain money
 * and can trade it all for an instrument or it has some instruments
 * and can trade them all for money.
 *
 * Using a time-series historical price data, this environment uses
 * power-of-two windows to the past, calculating maximum, minimum,
 * median and average historical prices over the window as ratios to
 * current price.
 *
 * This environment has three items to make it more realistic:
 *
 * * There is a spread of a certain percentage between the price
 *   quoted and the price received (or paid) by the agent, this is the
 *   spread of financial platform.
 * * For transactions resulting in financial gains, a percentage is left 
 *   aside to cover taxes.
 * * There is a minimum transaction price, if the agent ends up with 
 *   less than that, the game is lost.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Pablo Duboue
 */
class SimpleTrading implements Environment
{
    /**
     * Either sell or buy, depending on the position being held.
     * 
     * @var int
     */
    public const TRADE = 0;
    
    /**
     * Stay, i.e., do nothing.
     * 
     * @var int
     */
    public const STAY = 1;

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
     * Historical prices
     *
     * @var list<float>
     */
    
    protected array $historical;
    
    /**
     * Number of concentric windows (powers-of-two) to use.
     * 
     * @var int
     */
    protected int $windows;

    /**
     * Window sizes
     * 
     * @var list<int>
     */
    protected array $windowSizes;

    /**
     * Starting money
     * 
     * @var float
     */
    protected float $startingMoney;

    /**
     * Minimum transaction, in money units.
     * 
     * @var float
     */
    protected float $minimumTransaction;

    /**
     * Platform spread (percentage)
     * 
     * @var float
     */
    protected float $spread;

    /**
     * Capital gains tax (percentage over profit)
     * 
     * @var float
     */
    protected float $tax;

    /**
     * Reward for staying put (usually a small penalty)
     * 
     * @var float
     */
    protected float $stayReward;

    /**
     * Current time.
     * 
     * @var int
     */
    protected int $time;

    /**
     * Amount of money being held.
     * 
     * @var float
     */
    protected float $money;

    /**
     * Amount of financial instrument being held.
     * 
     * @var float
     */
    protected float $instrument;

    /**
     * Price paid for the financial instrument being held.
     *
     * @var float
     */
    protected float $pricePaid;
    
    /**
     * Create a new simple trading environment.
     *
     * @param list<float> $historicalPrices as a time series of floats
     * @param int $windows  number of windows in the past (as powers of two)
     * @param float $startingMoney starting money for trading
     * @param float $spread percentage of spread
     * @param float $tax percentage of capital gains tax
     * @param float $minimumTransaction minimum money amount to trade
     * @param float $stayReward reward to return for staying put
     */
    public function __construct(array $historicalPrices, int $windows, float $startingMoney, float $spread, float $tax,
                                float $minimumTransaction, float $stayReward)
    {
        //TODO validate all moneys are above 0 and all percentages are between 0 and 1
        $this->historical = $historicalPrices;
        $this->windows = $windows;
        $this->startingMoney = $startingMoney;
        $this->spread = $spread;
        $this->tax = $tax;
        $this->minimumTransaction = $minimumTransaction;
        $this->stayReward = $stayReward;

        $this->actionSpace = new ActionType(ActionType::DISCRETE, [ 'trade', 'stay' ]);
        $obs = [
            new ObservationType(ObservationType::DISCRETE, [ 0, # has money
                                                             1  # has instrument
            ]),
            new ObservationType(ObservationType::DISCRETE, [ 0, # under water
                                                             1  # sell for a profit
            ]),
            new ObservationType(ObservationType::DISCRETE, [ 0, # all good
                                                             1  # if sold, it'll go bankrupt
            ]),
            new ObservationType(ObservationType::CONTINUOUS, [ -99999, 99999 ]) # profit percentage
        ];
        $size = 1;
        $this->windowSizes = [];
        for($w = 0; $w<$windows; $w++){
            $size *= 2;
            $this->windowSizes[] = $size;
            $obs[] = new ObservationType(ObservationType::CONTINUOUS, [ -99999, 99999 ]); # minimum in window, as a percentage to current price
            $obs[] = new ObservationType(ObservationType::CONTINUOUS, [ -99999, 99999 ]); # maximum in window, as a percentage to current price
            $obs[] = new ObservationType(ObservationType::CONTINUOUS, [ -99999, 99999 ]); # average in window, as a percentage to current price
            $obs[] = new ObservationType(ObservationType::CONTINUOUS, [ -99999, 99999 ]); # mean in window, as a percentage to current price
        }
        $this->observationSpace = new ObservationType(ObservationType::COMPOSITE, $obs );
    }

    /**
     * Transforms current state into an observation.
     * 
     * @return Observation
     */
    protected function makeObservation() : Observation
    {
        $obs = [];
        $ifSold = $this->historical[$this->time] * (1.0-$this->spread) * $this->instrument;
        $profit = $ifSold - $this->pricePaid;
        $obs[] = new SimpleObservation($this->money > 0 ? 0 : 1,
                                       $this->observationSpace->params()[0]);
        $obs[] = new SimpleObservation($profit > 0 ? 1 : 0,
                                       $this->observationSpace->params()[1]);
        $obs[] = new SimpleObservation($this->instrument > 0 && $ifSold < $this->minimumTransaction ? 1 : 0,
                                       $this->observationSpace->params()[1]);
        $obs[] = new SimpleObservation($this->pricePaid > 0 ? $profit / $this->pricePaid : 0,
                                       $this->observationSpace->params()[1]);
        
        $current = $this->historical[$this->time];
        /*
        if($this->money > 0){
            $current *= 1.0 + $this->spread;
        }else{
            $current *= 1.0 - $this->spread;
        }
        */
        $max_size = $this->windowSizes[$this->windows-1];
        $max_window = array_slice($this->historical, $this->time - $max_size, $max_size);
        for($i=0; $i<$max_size; $i++) {
            $max_window[$i] /= $current;
        }
        $idx = 2;
        for($window=0; $window<$this->windows; $window++) {
            $size = $this->windowSizes[$window];
            $prices = array_slice($max_window, $max_size - $size, $size);
            sort($prices, SORT_NUMERIC);
            $min = $prices[0];
            $max = $prices[$size-1];
            $mean = ($prices[$size/2-1]+$prices[$size/2]) / 2;
            $avg = array_sum($prices) / $size;
            $obs[] = new SimpleObservation($min, $this->observationSpace->params()[$idx]);
            $obs[] = new SimpleObservation($max, $this->observationSpace->params()[$idx+1]);
            $obs[] = new SimpleObservation($avg, $this->observationSpace->params()[$idx+2]);
            $obs[] = new SimpleObservation($mean, $this->observationSpace->params()[$idx+3]);
            $idx+=4;
        }
        return new CompositeObservation($obs, $this->observationSpace);
    }
    
    /**
     * Indicates the end of an episode.
     *
     * @param ?int new time 
     * @param ?float new money
     * @param ?float new instrument
     * @param ?float new price paid
     * @return Response
     */
    public function reset(?int $time=null, ?float $money=null, ?float $instrument=null, ?float $pricePaid=null) : Response
    {
        $this->time = $time ?? $this->windowSizes[$this->windows-1];
        $this->money = $money ?? $this->startingMoney;
        $this->instrument = $instrument ?? 0.0;
        $this->pricePaid = $pricePaid ?? 0.0;
        
        return new Response($this->makeObservation(), 0.0, false);
    }

    /**
     * Run a step using the environment dynamics.
     *
     * @param \Rubix\ML\ReinforcementLearning\Action $action
     * @return \Rubix\ML\ReinforcementLearning\Response
     */
    public function step(Action $action): Response
    {
        $reward = 0.0;
        $done = false;
        switch($action->value()) {
        case self::TRADE:
            if($this->money > 0) { # buy
                $this->pricePaid = $this->money;
                $price = $this->historical[$this->time] * (1.0 + $this->spread);
                $this->instrument = $this->money / $price;
                $this->money = 0.0;
            }else{ # sell
                $price = $this->historical[$this->time] * (1.0 - $this->spread);
                $money = $this->instrument * $price;
                if($money > $this->pricePaid) {
                    $profit = $money - $this->pricePaid;
                    $money -= $profit * $this->tax;
                    $reward = 1000 * (1.0 + $profit / $this->pricePaid);
                }else{
                    $reward = -(($this->pricePaid - $money) / $this->pricePaid);
                }
                $this->money = $money;
                $this->instrument = 0.0;
                if($this->money < $this->minimumTransaction){
                    $done = true;
                }
            }
            break;
        case self::STAY:
            $reward = $this->stayReward;
            break;
        }
        $this->time++;
        if($this->time == count($this->historical) - 2){
            $done = true;
        }
        return new Response($this->makeObservation(), $reward, $done);
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
     * Current time in the enviroment
     *
     * @return int
     */
    public function time() : int
    {
        return $this->time;
    }

    /**
     * Current money held by the agent.
     *
     * @return float
     */
    public function money() : float
    {
        return $this->money;
    }

    /**
     * Current instrument held by the agent.
     *
     * @return float
     */
    public function instrument() : float
    {
        return $this->instrument;
    }

    /**
     * Price paid for the instrument held by the agent (zero if not
     * holding any instrument at the moment).
     *
     * @return float
     */
    public function pricePaid() : float
    {
        return $this->pricePaid;
    }

    /**
     * What the agent is worth
     *
     * @return float
     */
    public function worth() : float
    {
        if($this->money > 0) {
            return $this->money;
        }else{
            return $this->historical[$this->time] * (1.0 - $this->spread) * $this->instrument;
        }
    }

    /**
     * Prints the enviroment to the screen
     */
    public function show(?string $path = null) : void
    {
        $s  = "Time:      {$this->time}".PHP_EOL;
        $s .= "Price:     {$this->historical[$this->time]}".PHP_EOL;
        if($this->money > 0) {
            $s .= "Money:     {$this->money}".PHP_EOL;
        }else{
            $s .= "Instrument: {$this->instrument} (paid {$this->pricePaid} or " . ($this->pricePaid/$this->instrument) . " cost)".PHP_EOL;
        }
        
        if($path){
            file_put_contents($path, $s);
        }else{
            echo $s;
        }        
    }
}
