#pragma once

#include "ComputationalGraph/NodeBase.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
    class Optimizer;
    class InitializerBase;

    class Variable : public NodeBase
    {
    public:
        Variable(const Tensor& initValue, const string& name = "");
        Variable(const Shape& shape, InitializerBase* initializer = nullptr, const string& name = "");

        const string& Name() const { return m_Name; }

    private:
        string m_Name;
    };
}
