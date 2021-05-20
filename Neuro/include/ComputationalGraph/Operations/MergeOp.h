#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class NEURO_DLL_EXPORT MergeOp : public Operation
    {
    public:
        MergeOp(const vector<TensorLike*>& xs, EMergeMode mode, const string& name = "");

    protected:
        virtual void UpdateOutputShape() override;
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        EMergeMode m_Mode;
    };

    static Operation* merge_avg(const vector<TensorLike*>& xs, const string& name = "")
    {
        return new MergeOp(xs, AvgMerge, name);
    }

    static Operation* merge_min(const vector<TensorLike*>& xs, const string& name = "")
    {
        return new MergeOp(xs, MinMerge, name);
    }

    static Operation* merge_max(const vector<TensorLike*>& xs, const string& name = "")
    {
        return new MergeOp(xs, MaxMerge, name);
    }

    static Operation* merge_sum(const vector<TensorLike*>& xs, const string& name = "")
    {
        return new MergeOp(xs, SumMerge, name);
    }
}
