#pragma once

#include "ComputationalGraph/TensorLike.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
    class Optimizer;
    class InitializerBase;

    class Variable : public TensorLike
    {
    public:
        Variable(const Tensor& initValue, const string& name = "");
        Variable(const Shape& shape, InitializerBase* initializer = nullptr, const string& name = "");
    };
}
