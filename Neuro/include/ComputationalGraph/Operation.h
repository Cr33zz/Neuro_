#pragma once

#include "ComputationalGraph/TensorLike.h"

namespace Neuro
{
    class Tensor;

    class Operation : public TensorLike
    {
    public:
        virtual bool IsOp() const override { return true; }

        uint32_t LastComputeStep() const { return m_LastComputeStep; }
        vector<const Tensor*> GatherInputs() const;

        const Tensor& Compute(bool training);
        const vector<Tensor*>& ComputeGradient(const Tensor& grad);

        const vector<Tensor>& InputsGrads() const { return m_InputsGrads; }
        const vector<const Tensor*>& Inputs() const { return m_Inputs; }

        virtual bool CareAboutGradient() const override { return m_CareAboutGradient; }
        virtual void RefreshCareAboutGradient() override;
        virtual void OutputConsumed() override;
        virtual void InputGradConsumed(TensorLike* inputNode) override;
        
        virtual bool ForceAllocInputGradNode(size_t index) const { return false; }

        // Existence of training operations in fetched list will cause network to automatically run in training mode
        virtual bool IsTrainingOp() const { return false; }

    protected:
        Operation(const vector<TensorLike*>& inputNodes, const string& name);

        virtual void ComputeInternal() = 0;
        virtual void ComputeGradientInternal(const Tensor& grad) = 0;

        vector<const Tensor*> m_Inputs;
        vector<Tensor> m_InputsGrads;
        vector<Tensor*> m_InputsGradsPtrs; // for performance/convenience
        // This is used during gradient computation to figure out which consumers we care about.
        // We only care about computed ones in last forward pass
        uint32_t m_LastComputeStep = 0;
        /// Some operations like optimizer minimizations will consume outputs before computing gradients
        /// This flag in a hint for operation not to notify input nodes again
        bool m_InputsManuallyConsumed = false;
        bool m_CareAboutGradient = false;
        bool m_Training = false;
    };
}
