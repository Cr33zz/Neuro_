#pragma once

#include "ComputationalGraph/Constant.h"
#include "ComputationalGraph/Operations/Conv2DOp.h"
#include "ComputationalGraph/Operations/AbsOp.h"
#include "ComputationalGraph/Operations/SumOp.h"
#include "ComputationalGraph/Operations/AddOp.h"

namespace Neuro
{
    static Operation* total_variation(TensorLike* x, const string& name = "")
    {
        static Tensor horizKernel({ 1, -1, 0, 0, 0, 0,
                                    0, 0, 1, -1, 0, 0,
                                    0, 0, 0, 0, 1, -1 }, Shape(2, 1, 3, 3), "horiz_kernel");
        static Tensor vertKernel({ 1, -1, 0, 0, 0, 0,
                                   0, 0, 1, -1, 0, 0,
                                   0, 0, 0, 0, 1, -1 }, Shape(1, 2, 3, 3), "vert_kernel");

        auto horizDiff = conv2d(x, new Constant(horizKernel), 1, 0, NCHW, "horiz_diff");
        auto vertDiff = conv2d(x, new Constant(vertKernel), 1, 0, NCHW, "vert_diff");

        return add(sum(abs(horizDiff), _012Axes), sum(abs(vertDiff), _012Axes));
    }
}
