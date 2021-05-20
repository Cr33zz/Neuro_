#pragma once

#include <string>

#include "Tensors/Tensor.h"

#pragma warning(push)
#pragma warning(disable:4251)

namespace Neuro
{
    using namespace std;

    class Graph;
    class LayerBase;

    class NEURO_DLL_EXPORT TensorLike
    {
    public:
        virtual ~TensorLike() {}

        const Shape& GetShape() const { return m_Output.GetShape(); }

        const Tensor& Output() const { return m_Output; }
        Tensor& Output();
        const Tensor* OutputPtr() const { return &m_Output; }
        Tensor* OutputPtr();

        Tensor& OutputGrad() { return m_OutputGrad; }
        Tensor* OutputGradPtr() { return &m_OutputGrad; }

        const string& Name() const { return m_Name; }

        const vector<TensorLike*>& InputNodes() const { return m_InputNodes; }

        Graph* GetGraph() const { return m_Graph; }

        virtual bool IsOp() const { return false; }
        virtual bool IsPlaceholder() const { return false; }
        virtual bool IsVar() const { return false; }
        virtual bool IsConst() const { return false; }

        virtual bool ShouldPreload() const { return true; }
        virtual void Preload();
        virtual void PreloadForGradient();

        // Can be used to clean up internal state
        virtual void Reset() {}
        
        void AddInputNode(TensorLike* node) { m_InputNodes.push_back(node); }

        virtual bool CareAboutGradient() const;
        virtual void RefreshCareAboutGradient() {}
        virtual void OutputOnDeviceConsumed() {}
        virtual void InputGradConsumed(TensorLike* inputNode) {}

        // In most cases pre-loading nodes which don't care about gradient is pointless
        // However, in some cases we still need them to compute other inputs gradient (ie. multiplication)
        virtual bool ForcePreloadInputNode(size_t index) const { return false; }

        // This is relevant only for operations
        bool UndeterminedOutputShape() const { return m_UndeterminedOutputShape; }
        void SetAlwaysOffload(bool enabled) { m_AlwaysOffload = enabled; }
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
        // Normally batch size doesn't matter for output of operations in a graph; however, some operations (ie. transpose) 
        // can move it around, potentially causing validation/computation issues because initially batch size is unknown.
        // When output shape of any input is undetermined, operation will update it just before computing it's own output.
        // Undetermined output shape is causing all operations using it to be also undetermined.
        bool m_UndeterminedOutputShape : 1;
        bool m_AlwaysOffload : 1;
        bool m_Fetched : 1;

        friend class Operation;
        friend class Session;
        friend class Graph;
        friend class OptimizerBase;
    };
}

#pragma warning(pop)