#pragma once

#include <string>

#include "Tensors/Tensor.h"

namespace Neuro
{
    using namespace std;

    class Graph;

    class TensorLike
    {
    public:
        virtual ~TensorLike() {}

        const Shape& GetShape() const { return m_Output.GetShape(); }

        Tensor& Output() { return m_Output; }
        Tensor* OutputPtr() { return &m_Output; }

        Tensor& OutputGrad() { return m_OutputGrad; }
        Tensor* OutputGradPtr() { return &m_OutputGrad; }

        const string& Name() const { return m_Name; }

        Graph* GetGraph() const { return m_Graph; }

        virtual bool IsOp() const { return false; }
        virtual bool IsPlaceholder() const { return false; }
        virtual bool IsVar() const { return false; }

        void AddInputNode(TensorLike* node) { m_InputNodes.push_back(node); }

    protected:
        TensorLike(const string& name = "");

        Graph* m_Graph;
        vector<TensorLike*> m_Consumers;
        vector<TensorLike*> m_InputNodes;
        Tensor m_Output;
        Tensor m_OutputGrad;
        string m_Name;

        friend class Operation;
        friend class Session;
        friend class Graph;
        friend class OptimizerBase;
    };
}
