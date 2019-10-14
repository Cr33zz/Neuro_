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

        const Tensor& Compute(const vector<const Tensor*>& inputs);
        const vector<Tensor*>& ComputeGradient(const Tensor& grad);

        const vector<Tensor>& InputsGrads() const { return m_InputsGrads; }

        virtual void OutputConsumed() override;
        virtual void InputGradConsumed() override;

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
        size_t m_OutputConsumedCount = 0;
        size_t m_InputGradsConsumedCount = 0;
    };
}
