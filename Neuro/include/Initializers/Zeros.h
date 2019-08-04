#pragma once

#include "Initializers/InitializerBase.h"

namespace Neuro
{
    class Zeros : public InitializerBase
    {
	public:
        virtual void Init(Tensor& t, int fanIn, int fanOut) override;
	};
}
