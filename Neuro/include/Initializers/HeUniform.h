#pragma once

#include "Initializers/VarianceScaling.h"

namespace Neuro
{
    // He uniform variance scaling initializer.
    // It draws samples from a uniform distribution within[-limit, limit] where 'limit' is 'sqrt(6 / fan_in)'
    class HeUniform : public VarianceScaling
    {
    public:
        HeUniform() : VarianceScaling(2.f, FanIn, UniformDistribution) {}
    };
}
