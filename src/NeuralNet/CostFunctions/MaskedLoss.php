<?php

namespace Rubix\ML\NeuralNet\CostFunctions;

use Tensor\Matrix;
use Rubix\ML\Exceptions\InvalidArgumentException;

/**
 * Masked Loss
 *
 * If the output values have NANs, those columns are ignored for the
 * loss computation and its corresponding gradients are set to zero.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Pablo Duboue
 */
class MaskedLoss implements RegressionLoss
{
    /**
     * The underlining loss to use.
     *
     * @var RegressionLoss
     */
    protected RegressionLoss $loss;

    /**
     * @param RegressionLoss $loss
     */
    public function __construct(RegressionLoss $loss)
    {
        $this->loss = $loss;
    }

    /**
     * Fill-in the NANs from target matrix.
     *
     * @param Matrix $output the expected output with masks as NAN
     * @param Matrix $target the target output
     * @return Matrix with masked entries replaced by $target
     */
    protected function mask(Matrix $output, Matrix $target) : Matrix
    {
        $result = [];
        $o = $output->asArray();
        $t = $target->asArray();
        foreach($o as $r=>$row) {
            $res_row = [];
            foreach($row as $c=>$val){
                if(is_nan($val)) {
                    $res_row[] = $t[$r][$c];
                }else{
                    $res_row[] = $val;
                }
            }
            $result[] = $res_row;
        }
        return Matrix::quick($result);
    }

    /**
     * Compute the loss score.
     *
     * @internal
     *
     * @param \Tensor\Matrix $output
     * @param \Tensor\Matrix $target
     * @return float
     */
    public function compute(Matrix $output, Matrix $target) : float
    {
        
        return $this->loss->compute($this->mask($output, $target), $target);
    }

    /**
     * Calculate the gradient of the cost function with respect to the output.
     *
     * @internal
     *
     * @param \Tensor\Matrix $output
     * @param \Tensor\Matrix $target
     * @return \Tensor\Matrix
     */
    public function differentiate(Matrix $output, Matrix $target) : Matrix
    {
        return $this->loss->differentiate($this->mask($output, $target), $target);
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
        return "Masked Loss (underlining: {$this->loss})";
    }
}
