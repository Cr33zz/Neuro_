#pragma once

namespace Neuro
{
	class Tensor;

    class InitializerBase
    {
	public:
        virtual void Init(Tensor& t, int fanIn, int fanOut) = 0;
	};
}
