#pragma once

#include "Initializers/InitializerBase.h"

namespace Neuro
{
    class GlorotNormal : public InitializerBase
    {
	public:
        //It draws samples from a truncated normal distribution centered on 0 with deviation = sqrt(2 / (fanIn + fanOut))
        //where fanIn is the number of input units in the weight tensor and fanOut is the number of output units in the weight tensor.
        GlorotNormal(float gain = 1);

        static float NextSingle(int fanIn, int fanOut, float gain);

		virtual void Init(Tensor& t, int fanIn, int fanOut) override;

	private:
        float Gain;
	};
}
