#pragma once

#include "Initializers/InitializerBase.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT Const : public InitializerBase
    {
	public:
        Const(float value = 1.f);

        virtual void Init(Tensor& t) override;

	private:
        float m_Value;
	};
}
