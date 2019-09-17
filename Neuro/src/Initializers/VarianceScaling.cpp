#include <algorithm>

#include "Initializers/VarianceScaling.h"
#include "Initializers/Normal.h"
#include "Initializers/Uniform.h"
#include "Tensors/Tensor.h"
#include "Tensors/Shape.h"

namespace Neuro
{

    //////////////////////////////////////////////////////////////////////////
    VarianceScaling::VarianceScaling(float scale, EFanMode mode, EDistribution distribution)
        : m_Scale(scale), m_Mode(mode), m_Distribution(distribution)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void VarianceScaling::Init(Tensor& t)
    {
        auto fans = ComputeFans(t.GetShape());

        float fanIn = fans.first, fanOut = fans.second;
        float scale = m_Scale;

        if (m_Mode == FanIn)
            scale /= max(1.f, fanIn);
        else if (m_Mode == FanOut)
            scale /= max(1.f, fanOut);
        else
            scale /= max(1.f, (fanIn + fanOut) * 0.5f);

        if (m_Distribution == NormalDistribution)
        {
            float stddev = sqrt(scale) / 0.87962566103423978f;
            t.FillWithFunc([&]() { return Normal::NextTruncatedSingle(0.f, stddev); });
        }
        else
        {
            float limit = sqrt(3.f * scale);
            t.FillWithFunc([&]() { return Uniform::NextSingle(-limit, limit); });
        }
    }

    //////////////////////////////////////////////////////////////////////////
    pair<float, float> VarianceScaling::ComputeFans(const Shape& shape) const
    {
        float fanIn, fanOut;

        if (shape.NDim < 2)
        {
            fanIn = fanOut = (float)shape.Width();
        }
        else if (shape.NDim == 2)
        {
            fanIn = (float)shape.Width();
            fanOut = (float)shape.Height();
        }
        else if (shape.NDim >= 3 && shape.NDim <= 5)
        {
            // expects NCHW format
            uint32_t receptiveFieldSize = shape.Width() * shape.Height();
            fanIn = (float)shape.Depth() * receptiveFieldSize;
            fanOut = (float)shape.Batch() * receptiveFieldSize;
        }
        else
        {
            float product = 1;
            for (uint32_t i = 0; i < shape.NDim; ++i)
                product *= (float)shape.Dimensions[i];

            fanIn = sqrt(product);
            fanOut = sqrt(product);
        }

        return make_pair(fanIn, fanOut);
    }
}