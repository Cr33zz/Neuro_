#pragma once

#include "Initializers/InitializerBase.h"

namespace Neuro
{
    class Constant : public InitializerBase
    {
	public:
        Constant(float value = 1.f);

        virtual void Init(Tensor& t) override;

	private:
        float m_Value;
	};
}
