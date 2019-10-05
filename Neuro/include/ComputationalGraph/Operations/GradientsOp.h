#pragma once

#include "ComputationalGraph/Operation.h"

namespace Neuro
{
    class Variable;

    class GradientsOp : public Operation
    {
    public:
        GradientsOp(TensorLike* y, const vector<Variable*>& vars, const string& name = "");

        vector<TensorLike*> Grads() { return m_Grads; }

    protected:
        virtual void ComputeInternal() override;
        virtual void ComputeGradientInternal(const Tensor& grad) override { assert(false); }

    private:
        vector<Variable*> m_Vars;
        vector<TensorLike*> m_Grads;
        vector<TensorLike*> m_Order;
    };

    static vector<TensorLike*> gradients(TensorLike* y, const vector<Variable*>& vars, const string& name = "")
    {
        return (new GradientsOp(y, vars, name))->Grads();
    }

    static vector<TensorLike*> gradients(TensorLike* y, Variable* var, const string& name = "")
    {
        vector<Variable*> vars{ var };
        return gradients(y, vars, name);
    }
}
