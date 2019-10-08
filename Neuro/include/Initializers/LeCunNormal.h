#pragma once

#include "Initializers/VarianceScaling.h"

namespace Neuro
{
    // LeCun normal initializer.
    // It draws samples from a truncated normal distribution centered on 0 with 'stddev = sqrt(1 / fan_in)'
    class LeCunNormal : public VarianceScaling
    {
    public:
        LeCunNormal() : VarianceScaling(1.f, FanIn, NormalDistribution) {}
    };
}
