#pragma once

#include "CompGraph/NodeBase.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
    class Variable : public NodeBase
    {
    public:
        Variable(const Tensor& initValue)
        {
            m_Value = initValue;
            Graph.Default.Variables.Add(this);
        }

        const Tensor& Value() const { return m_Value;  }

    private:
        Tensor m_Value;
    };
}
