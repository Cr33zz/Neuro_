#pragma once

#include "ComputationalGraph/NameScope.h"
#include "ComputationalGraph/Operations/PowOp.h"
#include "ComputationalGraph/Operations/SumOp.h"
#include "ComputationalGraph/Operations/MultiplyOp.h"

namespace Neuro
{
    static Operation* l2_loss(TensorLike* x, const string& name = "")
    {
        NameScope scope("l2_loss");
        return multiply(sum(square(x)), 0.5f);
    }
}
