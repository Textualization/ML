<?php

namespace Rubix\ML\NeuralNet\Layers;

use Tensor\Matrix;
use Tensor\Vector;
use Tensor\ColumnVector;
use Rubix\ML\Deferred;
use Rubix\ML\Helpers\Params;
use Rubix\ML\NeuralNet\Parameter;
use Rubix\ML\NeuralNet\Initializers\He;
use Rubix\ML\NeuralNet\Optimizers\Optimizer;
use Rubix\ML\NeuralNet\Initializers\Constant;
use Rubix\ML\NeuralNet\Initializers\Initializer;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Generator;

/**
 * MultiTaskDense
 *
 * A subclass of Dense layers that can train one output neuron at a time.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Pablo Duboue
 */
class MultiTaskDense extends Dense
{
    /**
     * If defined, this layer will only output one value, the value
     * for its output neuron number $singleOutput.
     *
     * @var positive-int|null
     */
    protected ?int $singleOutput = null;

    /**
     * Full weights. Only used when $singleOutput is not null.
     *
     * @var \Rubix\ML\NeuralNet\Parameter|null
     */
    protected ?\Rubix\ML\NeuralNet\Parameter $fullWeights = null;

    /**
     * Full weights. Only used when $singleOutput is not null.
     *
     * @var \Rubix\ML\NeuralNet\Parameter|null
     */   
    protected ?\Rubix\ML\NeuralNet\Parameter $fullBiases = null;

    /**
     * Parameter ids for single output weights.
     * 
     * @var list<int>
     */
    protected array $singleOutputWeightIDs = [];
    
    /**
     * Parameter ids for single output biases.
     * 
     * @var list<int>
     */
    protected array $singleOutputBiasIDs = [];
    
    /**
     * @param int $neurons
     * @param float $l2Penalty
     * @param bool $bias
     * @param \Rubix\ML\NeuralNet\Initializers\Initializer|null $weightInitializer
     * @param \Rubix\ML\NeuralNet\Initializers\Initializer|null $biasInitializer
     * @throws \Rubix\ML\Exceptions\InvalidArgumentException
     */
    public function __construct(
        int $neurons,
        float $l2Penalty = 0.0,
        bool $bias = true,
        ?Initializer $weightInitializer = null,
        ?Initializer $biasInitializer = null
    ) {
        parent::__construct($neurons, $l2Penalty, $bias, $weightInitializer, $biasInitializer);
        $dummy = Vector::quick([0]);
        foreach (range(0, $neurons-1) as $singleOutput) {
            // reserve IDs for the single rows
            $param = new Parameter($dummy);
            $this->singleOutputWeightIDs[] = $param->id();
            $param = new Parameter($dummy);
            $this->singleOutputBiasIDs[] = $param->id();
        }
    }

    /**
     * Set single output (null to reset).
     *
     * @param ?int $singleOutput
     */
    public function singleOutput(?int $singleOutput)
    {
        if ($singleOutput === null) {
            // reset
            $weights = $this->fullWeights->param()->asArray();
            $weights[$this->singleOutput] = $this->weights->param()->rowAsVector(0)->asArray();
            $biases = $this->fullBiases->param()->asArray();
            $biases[$this->singleOutput] = $this->biases->param()->asArray()[0];
            $this->neurons = $this->fullWeights->param()->m();
            $this->weights = new Parameter(Matrix::quick($weights),       $this->fullWeights->id());
            $this->biases  = new Parameter(ColumnVector::quick($biases),  $this->fullBiases->id());
            $this->fullBiases   = null;
            $this->fullWeights  = null;
        } elseif ($singleOutput < 0 || $singleOutput > $this->neurons) {
            throw new InvalidArgumentException('Value of single ouput'
                . " must be greater than 0 and less than {$this->neurons}, $singleOuput given.");
        } else {
            // set single ouput
            $this->fullWeights = $this->weights;
            $this->fullBiases  = $this->biases;
            $this->neurons = 1;
            $this->weights = new Parameter($this->fullWeights->param()->rowAsVector($singleOutput)->asRowMatrix(),        $this->singleOutputWeightIDs[$singleOutput]);
            $this->biases  = new Parameter(ColumnVector::quick([ $this->fullBiases->param()->offsetGet($singleOutput) ]), $this->singleOutputBiasIDs[$singleOutput]);
        }
        $this->singleOutput = $singleOutput;
    }

    public function isSingleOutput() : bool
    {
        return $this->singleOutput !== null;
    }

    /**
     * Return the parameters of the layer.
     *
     * @internal
     *
     * @throws \Rubix\ML\Exceptions\RuntimeException
     * @return \Generator<\Rubix\ML\NeuralNet\Parameter>
     */
    public function parameters() : Generator
    {
        if (!$this->weights) {
            throw new RuntimeException('Layer has not been initialized.');
        }

        yield 'weights' => $this->weights;

        if ($this->biases) {
            yield 'biases' => $this->biases;
        }

        foreach(range(0, $this->neurons-1) as $single) {
            yield "weights$single" => new Parameter($this->weights->param()->rowAsVector($single)->asRowMatrix(),        $this->singleOutputWeightIDs[$single]);
            yield "bias$single"    => new Parameter(ColumnVector::quick([ $this->biases->param()->offsetGet($single) ]), $this->singleOutputBiasIDs[$single]);
        }
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return "MultiTaskDense (neurons: {$this->neurons}, l2 penalty: {$this->l2Penalty},"
            . ' bias: ' . Params::toString($this->bias) . ','
            . " weight initializer: {$this->weightInitializer},"
            . " bias initializer: {$this->biasInitializer},"
            . " single output: ". ($this->singleOutput??"full"). ")";
    }
}
