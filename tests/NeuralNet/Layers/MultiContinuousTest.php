<?php

namespace Rubix\ML\Tests\NeuralNet\Layers;

use Tensor\Matrix;
use Rubix\ML\Deferred;
use Rubix\ML\NeuralNet\Layers\Layer;
use Rubix\ML\NeuralNet\Layers\Output;
use Rubix\ML\NeuralNet\Layers\MultiContinuous;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;
use PHPUnit\Framework\TestCase;

/**
 * @group Layers
 * @covers \Rubix\ML\NeuralNet\Layers\MultiContinuous
 */
class MultiContinuousTest extends TestCase
{
    protected const RANDOM_SEED = 0;

    /**
     * @var \Tensor\Matrix
     */
    protected $input;

    /**
     * @var (list<int>|list<float>)[]
     */
    protected $labels;

    /**
     * @var \Rubix\ML\NeuralNet\Optimizers\Optimizer
     */
    protected $optimizer;

    /**
     * @var \Rubix\ML\NeuralNet\Layers\MultiContinuous
     */
    protected $layer;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->input = Matrix::quick([
            [2.5, 0.0, -6.0],
            [0.0, 0.0,  3.0]
        ]);

        $this->labels = [
            [0.0, 0.0], [-2.5, 1.0], [90, 10]
        ];

        $this->optimizer = new Stochastic(0.001);

        $this->layer = new MultiContinuous(new LeastSquares());

        srand(self::RANDOM_SEED);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(MultiContinuous::class, $this->layer);
        $this->assertInstanceOf(Output::class, $this->layer);
        $this->assertInstanceOf(Layer::class, $this->layer);
    }

    /**
     * @test
     */
    public function initializeForwardBackInfer() : void
    {
        $this->layer->initialize(2);

        $this->assertEquals(2, $this->layer->width());

        $expected = [
            [2.5, 0.0, -6.0],
            [0.0, 0.0,  3.0]
        ];

        $forward = $this->layer->forward($this->input);

        $this->assertInstanceOf(Matrix::class, $forward);
        $this->assertEquals($expected, $forward->asArray());

        [$computation, $loss] = $this->layer->back($this->labels, $this->optimizer);

        $this->assertInstanceOf(Deferred::class, $computation);
        $this->assertIsFloat($loss);

        $gradient = $computation->compute();

        $expected = [
            [0.8333333333333334, 0.8333333333333334, -32.0],
            [0.0, -0.33333333334, -2.3333333333333335]
        ];

        $this->assertInstanceOf(Matrix::class, $gradient);
        $this->assertEquals($expected, $gradient->asArray());

        $expected = [
            [2.5, 0.0, -6.0],
            [0.0, 0.0, 3.0]
        ];

        $infer = $this->layer->infer($this->input);
        
        $this->assertInstanceOf(Matrix::class, $infer);
        $this->assertEquals($expected, $infer->asArray());
    }
}
