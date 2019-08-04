#pragma once

#include "Initializers/InitializerBase.h"

namespace Neuro
{
    class Constant : public InitializerBase
    {
	public:
        Constant(float value = 1);

        virtual void Init(Tensor& t, int fanIn, int fanOut) override;

	private:
        float Value;
	};
}
