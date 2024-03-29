#pragma once

#include "ComputationalGraph/TensorLike.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
    class Optimizer;
    class InitializerBase;

    class NEURO_DLL_EXPORT Constant : public TensorLike
    {
    public:
        explicit Constant(const Tensor& value, const string& name = "");
        explicit Constant(float value, const string& name = "");

        virtual bool IsConst() const { return true; }

    private:
        static int s_NameId;
    };
}
