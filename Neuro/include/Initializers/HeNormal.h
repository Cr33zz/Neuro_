#pragma once

#include "Initializers/VarianceScaling.h"

namespace Neuro
{
    // He normal initializer.
    // It draws samples from a truncated normal distribution centered on 0 with 'stddev = sqrt(2 / fan_in)'
    class NEURO_DLL_EXPORT HeNormal : public VarianceScaling
    {
    public:
        HeNormal() : VarianceScaling(2.f, FanIn, NormalDistribution) {}
    };
}
