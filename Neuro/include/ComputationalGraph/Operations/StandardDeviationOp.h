#pragma once

#include "ComputationalGraph/NameScope.h"
#include "ComputationalGraph/Operation.h"
#include "ComputationalGraph/Operations/VarianceOp.h"
#include "ComputationalGraph/Operations/SqrtOp.h"

namespace Neuro
{
    static Operation* std_deviation(TensorLike* input, TensorLike* inputMean, EAxis axis = GlobalAxis, const string& name = "")
    {
        NameScope scope("std_deviation");
        return sqrt(variance(input, inputMean, axis));
    }
}
