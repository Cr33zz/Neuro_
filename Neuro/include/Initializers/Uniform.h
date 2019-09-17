#pragma once

#include "Initializers/InitializerBase.h"

namespace Neuro
{
    class Uniform : public InitializerBase
    {
	public:
        Uniform(float min = -0.05f, float max = 0.05f);

        static float NextSingle(float min, float max);

    protected:
        virtual void Init(Tensor& t) override;

	private:
        float m_Min;
        float m_Max;
	};
}
