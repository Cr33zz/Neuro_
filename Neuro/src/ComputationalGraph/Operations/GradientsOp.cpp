#include "ComputationalGraph/Operations/GradientsOp.h"
#include "ComputationalGraph/Variable.h"
#include "ComputationalGraph/Graph.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    GradientsOp::GradientsOp(TensorLike* y, vector<TensorLike*> params, const string& name)
        : Operation({ y }, name.empty() ? "gradients" : name), m_Params(params)
    {
        for (auto param : params)
        {
            m_Grads.push_back(new Variable(Tensor(param->Output().GetShape()).FillWithValue(0)));
            m_Grads.back()->AddInputNode(this); // build one-way connection required for forward pass
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void GradientsOp::ComputeInternal()
    {
        m_InputNodes[0]->GetGraph()->ComputeGradients(m_InputNodes/*, m_Params*/);
        for (size_t i = 0; i < m_Params.size(); ++i)
            m_Params[i]->OutputGrad().CopyTo(m_Grads[i]->Output());
    }
}