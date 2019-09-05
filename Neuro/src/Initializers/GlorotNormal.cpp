#include <algorithm>

#include "Initializers/GlorotNormal.h"
#include "Tensors/Tensor.h"
#include "Initializers/Normal.h"

namespace Neuro
{
	//////////////////////////////////////////////////////////////////////////
	GlorotNormal::GlorotNormal(float gain /*= 1*/)
	{
		Gain = gain;
	}

	//////////////////////////////////////////////////////////////////////////
	float GlorotNormal::NextSingle(int fanIn, int fanOut, float gain)
	{
		float scale = 1 / (float)max(1.f, (fanIn + fanOut) * 0.5f);
		float stdDev = gain * (float)sqrt(scale) / 0.87962566103423978f;
		return Normal::NextSingle(0, stdDev, 1);
	}

	//////////////////////////////////////////////////////////////////////////
	void GlorotNormal::Init(Tensor& t, int fanIn, int fanOut)
	{
        t.FillWithFunc([&]() { return NextSingle(fanIn, fanOut, Gain); });
	}
}
