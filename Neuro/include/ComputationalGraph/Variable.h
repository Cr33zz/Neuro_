#pragma once

#include "ComputationalGraph/NodeBase.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
    class Optimizer;

    class Variable : public NodeBase
    {
    public:
        Variable(const Tensor& initValue);

        Tensor& Value() { return m_Value;  }

    private:
        Tensor m_Value;
    };
}
