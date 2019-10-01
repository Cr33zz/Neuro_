#include "Loss.h"
#include "ComputationalGraph/Ops.h"

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
    NodeBase* BinaryCrossEntropy::Build(NodeBase* targetOutput, NodeBase* output)
    {
        return sum(sum(multiply(negative(targetOutput), log(output)), BatchAxis));
    }

    //////////////////////////////////////////////////////////////////////////
    NodeBase* MeanSquareError::Build(NodeBase* targetOutput, NodeBase* output)
    {
        return sum(mean(pow(subtract(output, targetOutput), 2)));
    }

	//////////////////////////////////////////////////////////////////////////
	Huber::Huber(float delta)
	{
		Delta = delta;
	}

    //////////////////////////////////////////////////////////////////////////
    Neuro::NodeBase* Huber::Build(NodeBase* targetOutput, NodeBase* output)
    {
        return ;
    }

    //////////////////////////////////////////////////////////////////////////
	void Huber::Compute(const Tensor& targetOutput, const Tensor& output, Tensor& result) const
	{
		targetOutput.Map([&](float yTrue, float y) { float a = y - yTrue; return abs(a) <= Delta ? (0.5f * a * a) : (Delta * (float)abs(a) - 0.5f * Delta * Delta); }, output, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Huber::Derivative(const Tensor& targetOutput, const Tensor& output, Tensor& result) const
	{
		targetOutput.Map([&](float yTrue, float y) { float a = y - yTrue; return abs(a) <= Delta ? a : (Delta * Sign(a)); }, output, result);
	}
}
