#pragma once

#include "ComputationalGraph/NodeBase.h"
#include "Tensors/Shape.h"

namespace Neuro
{
    class Placeholder : public NodeBase
    {
    public:
        Placeholder(const Shape& shape, const string& name = "");

        const Shape& GetShape() const { return m_Shape; }
        const string& Name() const { return m_Name; }

    private:
        Shape m_Shape;
        string m_Name;
    };
}
