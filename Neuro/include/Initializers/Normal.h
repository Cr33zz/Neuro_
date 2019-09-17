#pragma once

#include "Initializers/InitializerBase.h"

namespace Neuro
{
    class Normal : public InitializerBase
    {
	public:
        Normal(float mean = 0, float variance = 1, float scale = 1);

        static float NextSingle(float mean, float stdDeviation, float scale = 1.f);
        // Values whose magnitude is more than two standard deviations from the mean are dropped and re-picked.
        static float NextTruncatedSingle(float mean, float stdDeviation);

    protected:
        virtual void Init(Tensor& t) override;

	private:
        float m_Mean;
        float m_Variance;
        float m_Scale;

        static bool m_HasValue;
        static float m_Value;
	};
}
