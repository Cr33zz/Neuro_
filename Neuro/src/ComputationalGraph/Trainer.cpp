#include <map>
#include "ComputationalGraph/Trainer.h"
#include "ComputationalGraph/Session.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Trainer::Trainer(const vector<Placeholder*>& inputOps, const vector<Placeholder*>& targetOps, const vector<TensorLike*>& fetchOps)
    {
        m_InputOps = inputOps;
        m_TargetOps = targetOps;
        m_FetchOps = fetchOps;

        m_Order = Session::Default->BuildForwardOrder(m_FetchOps);
    }

    //////////////////////////////////////////////////////////////////////////
    tensor_ptr_vec_t Trainer::Train(const const_tensor_ptr_vec_t& inputs, const const_tensor_ptr_vec_t& outputs)
    {
        map<Placeholder*, const Tensor*> feeds;

        for (size_t i = 0; i < m_InputOps.size(); ++i)
            feeds[m_InputOps[i]] = inputs[i];
        for (size_t i = 0; i < m_TargetOps.size(); ++i)
            feeds[m_TargetOps[i]] = outputs[i];

        return Session::Default->RunInOrder(m_Order, m_FetchOps, feeds);
    }
}