#pragma once

#include "ComputationalGraph/NameScope.h"
#include "ComputationalGraph/Operation.h"
#include "ComputationalGraph/Operations/MeanOp.h"
#include "ComputationalGraph/Operations/PowOp.h"
#include "ComputationalGraph/Operations/SubtractOp.h"
#include "ComputationalGraph/Operations/SqrtOp.h"

namespace Neuro
{
    static Operation* variance(TensorLike* input, TensorLike* inputMean, EAxis axis = GlobalAxis, const string& name = "")
    {
        NameScope scope("variance");
        if (!inputMean)
            inputMean = mean(input, axis);
        auto xMu = sub(input, inputMean);
        return mean(square(xMu), axis);
    }
}
