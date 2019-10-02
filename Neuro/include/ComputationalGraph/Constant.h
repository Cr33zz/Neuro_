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
        Constant(const Tensor& value, const string& name = "");
        Constant(float value, const string& name = "");

        const string& Name() const { return m_Name; }

    private:
        string m_Name;
    };
}
