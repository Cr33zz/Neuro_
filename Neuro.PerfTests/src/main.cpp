#include "SimpleNetPerfTests.h"
#include "TensorPerfTests.h"
#include "ConvNetPeftTests.h"

int main()
{
	//SimpleNetPerfTests::Run();
    //TensorPerfTests::Run();

    Tensor input(Shape(27, 27, 2, 3)); input.FillWithRand();
    Tensor output = input.Pool(3, 2, Tensor::EPoolType::Max, Tensor::EPaddingType::Valid);
    Tensor outputGradient(output.GetShape()); outputGradient.FillWithRand();

    bool res = TestTools::VerifyActivationFuncDerivative(ELU(1), 3, Tensor::EOpMode::GPU);

	return 0;
}