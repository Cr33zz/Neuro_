﻿#pragma once

#include <string>

#include "Tensors/Tensor.h"

namespace Neuro
{
    using namespace std;

    class Graph;
    class LayerBase;

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

        const vector<TensorLike*>& InputNodes() const { return m_InputNodes; }

        Graph* GetGraph() const { return m_Graph; }

        virtual bool IsOp() const { return false; }
        virtual bool IsPlaceholder() const { return false; }
        virtual bool IsVar() const { return false; }
        virtual bool IsConst() const { return false; }

        virtual void Preload();
        virtual void PreloadForGradient();
        
        void AddInputNode(TensorLike* node) { m_InputNodes.push_back(node); }

        virtual bool CareAboutGradient() const;
        virtual void RefreshCareAboutGradient() {}
        virtual void OutputConsumed() {}
        virtual void InputGradConsumed(TensorLike* inputNode) {}

        // In most cases pre-loading nodes which don't care about gradient is pointless
        // However, in some cases we still need them to compute other inputs gradient (ie. multiplication)
        virtual bool ForcePreloadInputNode(size_t index) const { return false; }

        bool IsFetched() const { return m_Fetched; }
        void SetFetched(bool fetched) { m_Fetched = fetched; }

        struct metadata
        {
            LayerBase* layer; // layer which created this tensor
            size_t node_index; // node index in inbound nodes list in layer
            size_t tensor_index; // index in output tensors list
        };

        // Metadata is not used in case of pure computational graph networks but is required for layers abstraction
        metadata* m_Metadata = nullptr;

    protected:
        TensorLike(const string& name = "");

        Graph* m_Graph;
        vector<TensorLike*> m_Consumers;
        vector<TensorLike*> m_InputNodes;
        Tensor m_Output;
        Tensor m_OutputGrad;
        string m_Name;
        bool m_Fetched;

        friend class Operation;
        friend class Session;
        friend class Graph;
        friend class OptimizerBase;
    };
}
