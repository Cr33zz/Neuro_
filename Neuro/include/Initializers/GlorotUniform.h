#pragma once

#include "Initializers/VarianceScaling.h"

namespace Neuro
{
    // Glorot uniform initializer, also called Xavier uniform initializer.
    // It draws samples from a uniform distribution within[-limit, limit] where 'limit' is 'sqrt(6 / (fan_in + fan_out))
    class GlorotUniform : public VarianceScaling
    {
	public:
        GlorotUniform() : VarianceScaling(1.f, FanAvg, UniformDistribution) {}
	};
}
