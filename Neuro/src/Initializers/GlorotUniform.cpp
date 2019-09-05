#include <algorithm>

#include "Initializers/GlorotUniform.h"
#include "Tensors/Tensor.h"
#include "Initializers/Uniform.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	GlorotUniform::GlorotUniform(float gain)
	{
		Gain = gain;
	}

	//////////////////////////////////////////////////////////////////////////
	float GlorotUniform::NextSingle(int fanIn, int fanOut, float gain)
	{
		float scale = 1 / (float)max(1.0f, (fanIn + fanOut) * 0.5f);
		float limit = (float)sqrt(3 * scale);
		return Uniform::NextSingle(-limit, limit);
	}

	//////////////////////////////////////////////////////////////////////////
	void GlorotUniform::Init(Tensor& t, int fanIn, int fanOut)
	{
        t.FillWithFunc([&](){ return NextSingle(fanIn, fanOut, Gain); });
	}
}
