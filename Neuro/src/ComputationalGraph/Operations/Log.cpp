#include "ComputationalGraph/Operations/Log.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Op::Log::Log(NodeBase* x)
        : Operation({x})
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::Log::ComputeInternal()
    {
        m_Output.Resize(m_Inputs[0]->GetShape());
        m_Inputs[0]->Map([](float x) {return ::log(x); }, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::Log::ComputeGradientInternal(const Tensor& grad)
    {
        //in_grad = grad / x
        grad.Map([](float g, float x) {return g / x; }, *m_Inputs[0], m_InputsGrads[0]);
    }
}