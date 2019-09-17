#pragma once

#include <utility>

namespace Neuro
{
	class Tensor;
    class Shape;

    using namespace std;

    class InitializerBase
    {
	public:
        virtual void Init(Tensor& t);
        virtual void Init(Tensor& t, int fanIn, int fanOut) = 0;

    protected:
        pair<float, float> ComputeFans(const Shape& shape) const;
	};
}
