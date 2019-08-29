#include "Initializers/Uniform.h"
#include "Tensors/Tensor.h"
#include "Tools.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	Uniform::Uniform(float min, float max)
	{
		Min = min;
		Max = max;
	}

	//////////////////////////////////////////////////////////////////////////
	float Uniform::NextSingle(float min, float max)
	{
		return min + GlobalRng().NextFloat() * (max - min);
	}

	//////////////////////////////////////////////////////////////////////////
	void Uniform::Init(Tensor& t, int fanIn, int fanOut)
	{
		t.Map([&](float x) { return NextSingle(Min, Max); }, t);
	}
}
