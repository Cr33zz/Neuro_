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
    }

    //////////////////////////////////////////////////////////////////////////
    tensor_ptr_vec_t Trainer::Train(const const_tensor_ptr_vec_t& inputs, const const_tensor_ptr_vec_t& outputs)
    {
        //var init = tf.Graph.GetGlobalVariablesInitializer();
        //foreach (var op in init)
        //    session.Run(new Tensorflow.TF_Output[0], new Tensorflow.Tensor[0], new Tensorflow.TF_Output[0], new[] { op });

        map<Placeholder*, const Tensor*> feeds;

        for (size_t i = 0; i < m_InputOps.size(); ++i)
            feeds[m_InputOps[i]] = inputs[i];
        for (size_t i = 0; i < m_TargetOps.size(); ++i)
            feeds[m_TargetOps[i]] = outputs[i];

        return Session::Default->Run(m_FetchOps, feeds);
    }
}