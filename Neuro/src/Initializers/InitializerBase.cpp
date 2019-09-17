#include "Initializers\InitializerBase.h"
#include "Tensors/Shape.h"

namespace Neuro
{
    void InitializerBase::Init(Tensor& t)
    {

    }

    pair<float, float> InitializerBase::ComputeFans(const Shape& shape) const
    {
        float fanIn, fanOut;
        if (shape.NDim == 2)
        {
            fanIn = shape.Width;
            fanOut = shape.Height;
        }
        else if (shape.NDim >= 3 && shape.NDim <= 5)
        {
            // expects NCHW format
            int receptiveFieldSize = shape.Width * shape.Height;
            fanIn = shape.Depth * receptiveFieldSize;
            fanOut = shape.Batch * receptiveFieldSize;
        }
        else
        {
            float product = 1;
            for (int i = 0; i < shape.NDim; ++i)
                product *= shape.Dimensions[i];
            // No specific assumptions.
            fanIn = sqrt(product);
            fanOut = sqrt(product);
        }

        return make_pair(fanIn, fanOut);
    }
}