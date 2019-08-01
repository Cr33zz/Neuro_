#include "Tools.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
	const float Tools::_EPSILON = 10e-7f;

	//////////////////////////////////////////////////////////////////////////
	int Tools::AccNone(const Tensor& target, const Tensor& output)
	{
		return 0;
	}

	//////////////////////////////////////////////////////////////////////////
	int Tools::AccBinaryClassificationEquality(const Tensor& target, const Tensor& output)
	{
		int hits = 0;
		for (int n = 0; n < output.BatchSize; ++n)
			hits += target(0, 0, 0, n).Equals(roundf(output(0, 0, 0, n))) ? 1 : 0;
		return hits;
	}

	//////////////////////////////////////////////////////////////////////////
	int Tools::AccCategoricalClassificationEquality(const Tensor& target, const Tensor& output)
	{
		int hits = 0;
		for (int n = 0; n < output.BatchSize; ++n)
			hits += target.ArgMax(n).Equals(output.ArgMax(n)) ? 1 : 0;
		return hits;
	}

	//////////////////////////////////////////////////////////////////////////
	float Tools::Clip(float value, float min, float max)
	{
		return value < min ? min : (value > max ? max : value);
	}
}