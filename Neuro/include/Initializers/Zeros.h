#pragma once

#include "Initializers/InitializerBase.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT Zeros : public InitializerBase
    {
    public:
        virtual void Init(Tensor& t) override;
	};
}
