#pragma once

#include "Initializers/VarianceScaling.h"

namespace Neuro
{
    // LeCun uniform initializer.
    // It draws samples from a uniform distribution within[-limit, limit] where `limit` is `sqrt(3 / fan_in)`
    class LeCunUniform : public VarianceScaling
    {
    public:
        LeCunUniform() : VarianceScaling(1.f, FanIn, UniformDistribution) {}
    };
}
