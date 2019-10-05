#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class GradientsOp : public Operation
    {
    public:
        GradientsOp(TensorLike* y, vector<TensorLike*> params, const string& name = "");

        vector<TensorLike*> Grads() { return m_Grads; }

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override {}

    private:
        vector<TensorLike*> m_Params;
        vector<TensorLike*> m_Grads;
    };

    static vector<TensorLike*> gradients(TensorLike* y, vector<TensorLike*> params, const string& name = "")
    {
        return (new GradientsOp(y, params, name))->Grads();
    }

    static vector<TensorLike*> gradients(TensorLike* y, TensorLike* param, const string& name = "")
    {
        vector<TensorLike*> params{ param };
        return gradients(y, params, name);
    }
}
