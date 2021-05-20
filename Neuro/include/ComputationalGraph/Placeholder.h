#pragma once

#include "ComputationalGraph/TensorLike.h"
#include "Tensors/Shape.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT Placeholder : public TensorLike
    {
    public:
        explicit Placeholder(const Shape& shape, const string& name = "");
        explicit Placeholder(const Tensor& defaultVal, const string& name = "");

        virtual bool IsPlaceholder() const override { return true; }
    };
}
