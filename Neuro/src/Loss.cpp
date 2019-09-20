#include "Loss.h"
#include "Tensors/Tensor.h"
#include "Tools.h"

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
	void BinaryCrossEntropy::Compute(const Tensor& targetOutput, const Tensor& output, Tensor& result) const
	{
		Tensor clippedOutput = output.Clipped(_EPSILON, 1 - _EPSILON);
		targetOutput.Map([&](float yTrue, float y) { return -(yTrue * log(y) + (1 - yTrue) * log(1 - y)); }, clippedOutput, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void BinaryCrossEntropy::Derivative(const Tensor& targetOutput, const Tensor& output, Tensor& result) const
	{
		Tensor clippedOutput = output.Clipped(_EPSILON, 1 - _EPSILON);
		targetOutput.Map([&](float yTrue, float y) { return -yTrue / y + (1 - yTrue) / (1 - y); }, clippedOutput, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void MeanSquareError::Compute(const Tensor& targetOutput, const Tensor& output, Tensor& result) const
	{
		targetOutput.Map([&](float yTrue, float y) { return (y - yTrue) * (y - yTrue) * 0.5f; }, output, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void MeanSquareError::Derivative(const Tensor& targetOutput, const Tensor& output, Tensor& result) const
	{
		targetOutput.Map([&](float yTrue, float y) { return (y - yTrue); }, output, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Huber::Huber(float delta)
	{
		Delta = delta;
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
