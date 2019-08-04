#pragma once

#include "Initializers/InitializerBase.h"

namespace Neuro
{
    class Normal : public InitializerBase
    {
	public:
        Normal(float mean = 0, float variance = 1, float scale = 1);

        static float NextSingle(float mean, float stdDeviation, float scale);

		virtual void Init(Tensor& t, int fanIn, int fanOut) override;

	private:
        float Mean;
        float Variance;
        float Scale;

        static bool HasValue;
        static float Value;
	};
}
