#include "SimpleNetPerfTests.h"
#include "TensorPerfTests.h"
#include "ConvNetPeftTests.h"

int main()
{
	//SimpleNetPerfTests::Run();
    //TensorPerfTests::Run();

    Tensor input(Shape(27, 27, 2, 3)); input.FillWithRand();
    Tensor output = input.Pool(3, 2, EPoolingMode::Max, EPaddingMode::Valid);
    Tensor outputGradient(output.GetShape()); outputGradient.FillWithRand();

    bool res = TestTools::VerifyActivationFuncDerivative(ELU(1), 3, EOpMode::GPU);

	return 0;
}