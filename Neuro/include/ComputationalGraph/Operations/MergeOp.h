#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class MergeOp : public Operation
    {
    public:
        MergeOp(const vector<TensorLike*>& xs, EMergeMode mode, const string& name = "");

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override;

    private:
        EMergeMode m_Mode;
    };

    static Operation* merge_avg(const vector<TensorLike*>& xs, const string& name = "")
    {
        return new MergeOp(xs, MergeAvg, name);
    }

    static Operation* merge_min(const vector<TensorLike*>& xs, const string& name = "")
    {
        return new MergeOp(xs, MergeMin, name);
    }

    static Operation* merge_max(const vector<TensorLike*>& xs, const string& name = "")
    {
        return new MergeOp(xs, MergeMax, name);
    }

    static Operation* merge_sum(const vector<TensorLike*>& xs, const string& name = "")
    {
        return new MergeOp(xs, MergeSum, name);
    }
}
