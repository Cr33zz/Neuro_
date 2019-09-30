#pragma once

#include "Initializers/InitializerBase.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
    class Uniform : public InitializerBase
    {
	public:
        Uniform(float min = -0.05f, float max = 0.05f);

        static float NextSingle(float min, float max);
        static Tensor Random(float min, float max, const Shape& shape);

    protected:
        virtual void Init(Tensor& t) override;

	private:
        float m_Min;
        float m_Max;
	};
}
