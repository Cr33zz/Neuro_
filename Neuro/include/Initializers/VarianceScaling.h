#pragma once

#include <utility>

#include "Initializers/InitializerBase.h"

namespace Neuro
{
    class Shape;
    class Tensor;

    using namespace std;

    enum EFanMode
    {
        FanIn,
        FanOut,
        FanAvg,
    };

    enum EDistribution
    {
        NormalDistribution,
        UniformDistribution
    };

    class VarianceScaling : public InitializerBase
    {
    public:
        VarianceScaling(float scale = 1, EFanMode mode = FanIn, EDistribution distribution = NormalDistribution);

        virtual void Init(Tensor& t) override;

    private:
        pair<float, float> ComputeFans(const Shape& shape) const;

        EFanMode m_Mode;
        EDistribution m_Distribution;
        float m_Scale;
    };
}
