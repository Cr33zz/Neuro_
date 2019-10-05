#include "ComputationalGraph/Operations/NegativeOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    NegativeOp::NegativeOp(TensorLike* x, const string& name)
        : Operation({ x }, name.empty() ? "negative" : name)
    {
        m_Output.Resize(x->GetShape());
    }

    //////////////////////////////////////////////////////////////////////////
    void NegativeOp::ComputeInternal()
    {
        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_Inputs[0]->Map([](float x) {return -x; }, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void NegativeOp::ComputeGradientInternal(const Tensor& grad)
    {
        //in_grad = -grad
        grad.Map([](float g) {return -g; }, m_InputsGrads[0]);
    }
}