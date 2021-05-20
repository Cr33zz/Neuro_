#pragma once

#include "Types.h"

namespace Neuro
{
	class Tensor;

    class NEURO_DLL_EXPORT InitializerBase
    {
	public:
        virtual void Init(Tensor& t) = 0;
	};
}
