#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class GradientsOp : public Operation
    {
    public:
        GradientsOp(TensorLike* y, vector<TensorLike*> params);

        vector<TensorLike*> Grads() { return m_Grads; }

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override {}

    private:
        vector<TensorLike*> m_Params;
        vector<TensorLike*> m_Grads;
    };

    static vector<TensorLike*> gradients(TensorLike* y, vector<TensorLike*> params)
    {
        return (new GradientsOp(y, params))->Grads();
    }

    static vector<TensorLike*> gradients(TensorLike* y, TensorLike* param)
    {
        vector<TensorLike*> params{ param };
        return gradients(y, params);
    }
}
