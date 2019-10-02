#include <map>
#include "ComputationalGraph/Predicter.h"
#include "ComputationalGraph/Session.h"

namespace Neuro
{

    //////////////////////////////////////////////////////////////////////////
    Predicter::Predicter(const vector<Placeholder*>& inputOps, const vector<TensorLike*>& outputOps)
    {
        m_InputOps = inputOps;
        m_OutputOps = outputOps;
    }

    //////////////////////////////////////////////////////////////////////////
    tensor_ptr_vec_t Predicter::Predict(const const_tensor_ptr_vec_t& inputs)
    {
        //var init = tf.Graph.GetGlobalVariablesInitializer();
        //foreach (var op in init)
        //    session.Run(new Tensorflow.TF_Output[0], new Tensorflow.Tensor[0], new Tensorflow.TF_Output[0], new[] { op });

        map<Placeholder*, const Tensor*> feeds;

        for (size_t i = 0; i < m_InputOps.size(); ++i)
            feeds[m_InputOps[i]] = inputs[i];

        return Session::Default->Run(m_OutputOps, feeds);
    }
}