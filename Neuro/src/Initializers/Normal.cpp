#include "Initializers/Normal.h"
#include "Tensors/Tensor.h"
#include "Tools.h"

namespace Neuro
{
	bool Normal::HasValue = false;
	float Normal::Value = 0;

	//////////////////////////////////////////////////////////////////////////
	Normal::Normal(float mean /*= 0*/, float variance /*= 1*/, float scale /*= 1*/)
	{
		Mean = mean;
		Variance = variance;
		Scale = scale;
	}

	//////////////////////////////////////////////////////////////////////////
	float Normal::NextSingle(float mean, float stdDeviation, float scale)
	{
		//based upon https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/randomkit.c
		float variance = stdDeviation * stdDeviation;

		if (HasValue)
		{
			HasValue = false;
			return (variance * (Value)+mean) * scale;
		}

		float x1, x2, r2;
		do
		{
			x1 = 2 * GlobalRng().NextFloat() - 1;
			x2 = 2 * GlobalRng().NextFloat() - 1;
			r2 = x1 * x1 + x2 * x2;
		}

		while (r2 >= 1.0 || r2 == 0.0);

		//Polar method, a more efficient version of the Box-Muller approach.
		float f = (float)sqrt(-2 * log(r2) / r2);

		HasValue = true;
		Value = f * x1;

		return (variance * (f * x2) + mean) * scale;
	}

	//////////////////////////////////////////////////////////////////////////
	void Normal::Init(Tensor& t, int fanIn, int fanOut)
	{
		t.Map([&](float x) { return NextSingle(Mean, Variance, Scale); }, t);
	}
}
