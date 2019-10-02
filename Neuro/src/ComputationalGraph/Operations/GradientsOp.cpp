#include "ComputationalGraph/Operations/GradientsOp.h"
#include "ComputationalGraph/Variable.h"
#include "ComputationalGraph/Session.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    GradientsOp::GradientsOp(TensorLike* y, vector<TensorLike*> params)
        : Operation({y}), m_Params(params)
    {
        for (auto param : params)
        {
            m_Grads.push_back(new Variable(Tensor(param->Output().GetShape()).FillWithValue(0)));
            m_Grads.back()->AddInputNode(this);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void GradientsOp::ComputeInternal()
    {
        Session::Default->ComputeGradients(m_InputNodes[0]/*, m_Params*/);
        for (size_t i = 0; i < m_Params.size(); ++i)
            m_Params[i]->OutputGrad().CopyTo(m_Grads[i]->Output());
    }
}