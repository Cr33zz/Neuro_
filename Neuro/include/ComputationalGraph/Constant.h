#pragma once

#include "ComputationalGraph/TensorLike.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
    class Optimizer;
    class InitializerBase;

    class Constant : public TensorLike
    {
    public:
        explicit Constant(const Tensor& value, const string& name = "");
        explicit Constant(float value, const string& name = "");
    };
}
