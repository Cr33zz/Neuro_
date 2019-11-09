#pragma once

#include "ComputationalGraph/NameScope.h"
#include "ComputationalGraph/Operation.h"
#include "ComputationalGraph/Operations/MeanOp.h"
#include "ComputationalGraph/Operations/PowOp.h"
#include "ComputationalGraph/Operations/SubtractOp.h"
#include "ComputationalGraph/Operations/SqrtOp.h"

namespace Neuro
{
    static Operation* variance(TensorLike* input, TensorLike* inputMean = nullptr, EAxis axis = GlobalAxis, const string& name = "")
    {
        NameScope scope(name.empty() ? "variance" : name);
        if (!inputMean)
            inputMean = mean(input, axis, "input_mean");
        return mean(square(sub(input, inputMean)), axis);
    }
}
