#pragma once

#include "ComputationalGraph/NodeBase.h"
#include "Tensors/Shape.h"

namespace Neuro
{
    class Placeholder : public NodeBase
    {
    public:
        Placeholder(const Shape& shape);
        const Shape& GetShape() const { return m_Shape; }

    private:
        Shape m_Shape;
    };
}
