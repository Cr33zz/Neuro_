#pragma once

#include <string>

#include "Tensors/Tensor.h"

namespace Neuro
{
    using namespace std;

    class Operation;

    class NodeBase
    {
    public:
        const Tensor& Output() const { return m_Output; }
        Tensor* OutputPtr() { return &m_Output; }
        const string& Name() const { return m_Name; }

        virtual bool IsOp() const { return false; }

    protected:
        vector<NodeBase*> m_Consumers;
        vector<NodeBase*> m_InputNodes;
        Tensor m_Output;
        string m_Name;

        friend class Operation;
        friend class Session;
        friend class Optimizer;
    };
}
