#pragma once

#include "ComputationalGraph/NameScope.h"
#include "ComputationalGraph/Operations/PowOp.h"
#include "ComputationalGraph/Operations/MeanOp.h"
#include "ComputationalGraph/Operations/SubtractOp.h"

namespace Neuro
{
    static Operation* mse(TensorLike* yTrue, TensorLike* yPred, const string& name = "")
    {
        NameScope scope("mse");
        return mean(square(sub(yPred, yTrue)));
    }
}
