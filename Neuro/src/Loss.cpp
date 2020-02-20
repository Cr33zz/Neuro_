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
        return mean(negative(add(multiply(targetOutput, log(clippedOutput, "log(y)"), "yTrue×log(y)"),
                                 multiply(subtract(new Constant(1), targetOutput, "1-yTrue"), 
                                          log(subtract(new Constant(1), clippedOutput, "1-y"), "log(1-y)"), "(1-yTrue)×log(1-y)"), "yTrue×log(y)+(1-yTrue)×log(1-y)")));
    }

    //////////////////////////////////////////////////////////////////////////
    TensorLike* MeanSquareError::Build(TensorLike* targetOutput, TensorLike* output)
    {
        NameScope scope("mean_square_error");
        return mean(square(subtract(output, targetOutput)));
    }

    //////////////////////////////////////////////////////////////////////////
    TensorLike* MeanAbsoluteError::Build(TensorLike* targetOutput, TensorLike* output)
    {
        NameScope scope("mean_absolute_error");
        return mean(abs(subtract(output, targetOutput)));
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
