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
        virtual ~NodeBase() {}

        const Tensor& Output() const { return m_Output; }
        Tensor& Output() { return m_Output; }
        Tensor* OutputPtr() { return &m_Output; }

        const Tensor& OutputGrad() const { return m_OutputGrad; }
        Tensor* OutputGradPtr() { return &m_OutputGrad; }

        const string& Name() const { return m_Name; }

        virtual bool IsOp() const { return false; }

    protected:
        vector<NodeBase*> m_Consumers;
        vector<NodeBase*> m_InputNodes;
        Tensor m_Output;
        Tensor m_OutputGrad;
        string m_Name;

        friend class Operation;
        friend class Session;
        friend class Optimizer;
    };
}
