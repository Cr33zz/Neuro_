#include "SimpleNetPerfTests.h"
#include "TensorPerfTests.h"
#include "ConvNetPeftTests.h"

int main()
{
	//SimpleNetPerfTests::Run();
    //TensorPerfTests::Run();

    Tensor::SetDefaultOpMode(EOpMode::CPU);
    Tensor input(Shape(2, 2, 3, 3)); input.FillWithRand();
    Tensor gamma(Shape(2, 2, 3, 1)); gamma.FillWithRand();
    Tensor beta(Shape(2, 2, 3, 1)); beta.FillWithRand();
    Tensor runningMean(Shape(2, 2, 3, 1)); runningMean.FillWithRand();
    Tensor runningVariance(Shape(2, 2, 3, 1)); runningVariance.FillWithRand(-1, 0, 1);

    Tensor result(input.GetShape());
    input.BatchNormalization(gamma, beta, runningMean, runningVariance, result);

    Tensor::SetDefaultOpMode(EOpMode::CPU);
    input.SetOpMode(EOpMode::GPU);
    Tensor result2(input.GetShape());
    input.BatchNormalization(gamma, beta, runningMean, runningVariance, result2);

    bool eq = result.Equals(result2);


	return 0;
}