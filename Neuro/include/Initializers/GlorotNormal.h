#pragma once

#include "Initializers/VarianceScaling.h"

namespace Neuro
{
    // Glorot normal initializer, also called Xavier normal initializer.
    // It draws samples from a truncated normal distribution centered on 0 with deviation = sqrt(2 / (fanIn + fanOut))
    class NEURO_DLL_EXPORT GlorotNormal : public VarianceScaling
    {
	public:
        GlorotNormal() : VarianceScaling(1.f, FanAvg, NormalDistribution) {}
	};
}
