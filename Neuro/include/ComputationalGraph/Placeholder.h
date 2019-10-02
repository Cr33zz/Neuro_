#pragma once

#include "ComputationalGraph/TensorLike.h"
#include "Tensors/Shape.h"

namespace Neuro
{
    class Placeholder : public TensorLike
    {
    public:
        Placeholder(const Shape& shape, const string& name = "");

        virtual bool IsPlaceholder() const override { return true; }

        const Shape& GetShape() const { return m_Shape; }

    private:
        Shape m_Shape;
    };
}
