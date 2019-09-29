#pragma once

#include "CompGraph/NodeBase.h"
#include "Tensors/Shape.h"

namespace Neuro
{
    class Placeholder : public NodeBase
    {
    public:
        Placeholder(const Shape& shape)
        {
            m_Shape = shape;
            Graph.Default.Placeholders.Add(this);
        }

    private:
        Shape m_Shape;
    };
}
