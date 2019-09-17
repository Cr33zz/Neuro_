#pragma once

namespace Neuro
{
	class Tensor;

    class InitializerBase
    {
	public:
        virtual void Init(Tensor& t) = 0;
	};
}
