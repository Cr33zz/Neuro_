#include "Loss.h"
#include "Tensors/Tensor.h"
#include "Tools.h"

namespace Neuro
{

	//////////////////////////////////////////////////////////////////////////
	void CategoricalCrossEntropy::Compute(const Tensor& targetOutput, const Tensor& output, Tensor& result)
	{
		Tensor clippedOutput = output.Clipped(Tools::_EPSILON, 1 - Tools::_EPSILON);
		targetOutput.Map([&](float yTrue, float y) { return -yTrue * (float)log(y); }, clippedOutput, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void CategoricalCrossEntropy::Derivative(const Tensor& targetOutput, const Tensor& output, Tensor& result)
	{
		Tensor clippedOutput = output.Clipped(Tools::_EPSILON, 1 - Tools::_EPSILON);
		targetOutput.Map([&](float yTrue, float y) { return -yTrue / y; }, clippedOutput, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void CrossEntropy::Compute(const Tensor& targetOutput, const Tensor& output, Tensor& result)
	{
		Tensor clippedOutput = output.Clipped(Tools::_EPSILON, 1 - Tools::_EPSILON);
		targetOutput.Map([&](float yTrue, float y) { return -yTrue * (float)log(y) - (1 - yTrue) * (float)log(1 - y); }, clippedOutput, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void CrossEntropy::Derivative(const Tensor& targetOutput, const Tensor& output, Tensor& result)
	{
		Tensor clippedOutput = output.Clipped(Tools::_EPSILON, 1 - Tools::_EPSILON);
		targetOutput.Map([&](float yTrue, float y) { return -yTrue / y + (1 - yTrue) / (1 - y); }, clippedOutput, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void MeanSquareError::Compute(const Tensor& targetOutput, const Tensor& output, Tensor& result)
	{
		targetOutput.Map([&](float yTrue, float y) { return (y - yTrue) * (y - yTrue); }, output, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void MeanSquareError::Derivative(const Tensor& targetOutput, const Tensor& output, Tensor& result)
	{
		targetOutput.Map([&](float yTrue, float y) { return (y - yTrue); }, output, result);
	}

	//////////////////////////////////////////////////////////////////////////
	Huber::Huber(float delta)
	{
		Delta = delta;
	}

	//////////////////////////////////////////////////////////////////////////
	void Huber::Compute(const Tensor& targetOutput, const Tensor& output, Tensor& result)
	{
		targetOutput.Map([&](float yTrue, float y) { float a = y - yTrue; return abs(a) <= Delta ? (0.5f * a * a) : (Delta * (float)abs(a) - 0.5f * Delta * Delta); }, output, result);
	}

	//////////////////////////////////////////////////////////////////////////
	void Huber::Derivative(const Tensor& targetOutput, const Tensor& output, Tensor& result)
	{
		targetOutput.Map([&](float yTrue, float y) { float a = y - yTrue; return abs(a) <= Delta ? a : (Delta * Tools::Sign(a)); }, output, result);
	}

	/*CategoricalCrossEntropy Loss::CategoricalCrossEntropyLoss = CategoricalCrossEntropy();
	CrossEntropy Loss::CrossEntropyLoss = CrossEntropy();
	MeanSquareError Loss::MeanSquareErrorLoss = MeanSquareError();
	Huber Loss::Huber1Loss = Huber(1);*/
}
