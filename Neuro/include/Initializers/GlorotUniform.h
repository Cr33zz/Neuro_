#pragma once

#include "Initializers/InitializerBase.h"

namespace Neuro
{
    class GlorotUniform : public InitializerBase
    {
	public:
        GlorotUniform(float gain = 1);

        static float NextSingle(int fanIn, int fanOut, float gain);

		virtual void Init(Tensor& t, int fanIn, int fanOut) override;

	private:
        float Gain;
	};
}
