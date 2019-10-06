#include "ComputationalGraph/Operations/GradientsOp.h"
#include "ComputationalGraph/Variable.h"
#include "ComputationalGraph/Graph.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    GradientsOp::GradientsOp(TensorLike* y, const vector<Variable*>& params, const string& name)
        : Operation({ y }, name.empty() ? "gradients" : name), m_Vars(params)
    {
        m_Order = Graph::Default()->BuildBackwardOrder({ y }, params);
        for (auto param : params)
        {
            m_Grads.push_back(new Variable(zeros(param->Output().GetShape()), param->Name() + "_grad"));
            m_Grads.back()->AddInputNode(this); // build one-way connection required for forward pass
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void GradientsOp::ComputeInternal()
    {
        m_InputNodes[0]->GetGraph()->ComputeGradientsInOrder(m_Order, m_Vars);
        for (size_t i = 0; i < m_Vars.size(); ++i)
            m_Vars[i]->OutputGrad().CopyTo(m_Grads[i]->Output());
    }
}