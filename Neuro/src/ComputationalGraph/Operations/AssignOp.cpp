#include "ComputationalGraph/Operations/AssignOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    AssignOp::AssignOp(TensorLike* x, TensorLike* val, const string& name)
        : Operation({ x, val }, name.empty() ? "assign" : name)
    {
        assert(x->GetShape() == val->GetShape());
    }

    //////////////////////////////////////////////////////////////////////////
    void AssignOp::ComputeInternal()
    {
        if (m_InputNodes[0]->CareAboutGradient())
            m_Inputs[1]->CopyTo(m_InputNodes[0]->Output());
    }
}