#include "Loss.h"
#include "Tensors/Tensor.h"
#include "Tools.h"
#include "ComputationalGraph/Ops.h"
#include "ComputationalGraph/Constant.h"
#include "ComputationalGraph/NameScope.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	//void CategoricalCrossEntropy::Compute(const Tensor& targetOutput, const Tensor& output, Tensor& result) const
	//{
	//	Tensor clippedOutput = output.Clipped(_EPSILON, 1 - _EPSILON);
	//	targetOutput.Map([&](float yTrue, float y) { return -yTrue * log(y); }, clippedOutput, result);
	//}

	////////////////////////////////////////////////////////////////////////////
	//void CategoricalCrossEntropy::Derivative(const Tensor& targetOutput, const Tensor& output, Tensor& result) const
	//{
	//	Tensor clippedOutput = output.Clipped(_EPSILON, 1 - _EPSILON);
	//	targetOutput.Map([&](float yTrue, float y) { return -yTrue / y; }, clippedOutput, result);
	//}

	//////////////////////////////////////////////////////////////////////////
    TensorLike* BinaryCrossEntropy::Build(TensorLike* targetOutput, TensorLike* output)
    {
        NameScope scope("cross_entropy");
        auto clippedOutput = clip(output, _EPSILON, 1 - _EPSILON);
        return negative(add(multiply(targetOutput, log(clippedOutput)),
                            multiply(subtract(new Constant(1), targetOutput), log(subtract(new Constant(1), clippedOutput)))));
    }

    //////////////////////////////////////////////////////////////////////////
    TensorLike* MeanSquareError::Build(TensorLike* targetOutput, TensorLike* output)
    {
        NameScope scope("mean_square_error");
        return multiply(square(subtract(output, targetOutput)), new Constant(0.5f));
    }

	//////////////////////////////////////////////////////////////////////////
	Huber::Huber(float delta)
	{
		Delta = delta;
	}

    //////////////////////////////////////////////////////////////////////////
    Neuro::TensorLike* Huber::Build(TensorLike* targetOutput, TensorLike* output)
    {
        NameScope scope("huber");
        assert(false);
        return nullptr;//huber(targetOutput, output);
    }
}
