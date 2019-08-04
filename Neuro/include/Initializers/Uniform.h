#pragma once

#include "Initializers/InitializerBase.h"

namespace Neuro
{
    class Uniform : public InitializerBase
    {
	public:
        Uniform(float min = -0.05f, float max = 0.05f);

        static float NextSingle(float min, float max);

		virtual void Init(Tensor& t, int fanIn, int fanOut) override;

	private:
        float Min;
        float Max;
	};
}
