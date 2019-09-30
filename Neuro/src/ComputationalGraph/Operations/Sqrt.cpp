#include "ComputationalGraph/Operations/Sqrt.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Op::Sqrt::Sqrt(NodeBase* x)
        : Operation({ x })
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::Sqrt::ComputeInternal()
    {
        m_Inputs[0]->Map([](float x) {return ::sqrt(x); }, m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Op::Sqrt::ComputeGradientInternal(const Tensor& grad)
    {
        //in_grad = 1 / (2 * sqrt(x))
        grad.Map([](float g, float x) {return g / 2 * x; }, m_Output, m_InputsGrads[0]);
    }
}