#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT AccuracyOp : public Operation
    {
    public:
        AccuracyOp(TensorLike* target, TensorLike* output, const string& name = "");

    protected:
        virtual void UpdateOutputShape() override;
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override { assert(false); }
    };

    class NEURO_DLL_EXPORT BinaryAccuracyOp : public Operation
    {
    public:
        BinaryAccuracyOp(TensorLike* target, TensorLike* output, const string& name = "");

    protected:
        virtual void UpdateOutputShape() override;
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override { assert(false); }
    };

    static Operation* accuracy(TensorLike* target, TensorLike* output, const string& name = "")
    {
        return new AccuracyOp(target, output, name);
    }

    static Operation* binary_accuracy(TensorLike* target, TensorLike* output, const string& name = "")
    {
        return new BinaryAccuracyOp(target, output, name);
    }
}
