#include <map>

#include "ComputationalGraph/Predicter.h"
#include "ComputationalGraph/Session.h"
#include "ComputationalGraph/Graph.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Predicter::Predicter(const vector<Placeholder*>& inputPlaceholders, const vector<TensorLike*>& outputOps)
    {
        m_InputPlaceholders = inputPlaceholders;
        m_OutputOps = outputOps;

        bool isTraining = Graph::Default()->BuildForwardOrder(m_OutputOps, m_Order);

        NEURO_ASSERT(!isTraining, "Fetching training operation in predictor.");

        for (size_t i = 0; i < m_InputPlaceholders.size(); ++i)
            m_Feeds[m_InputPlaceholders[i]] = nullptr;
    }

    //////////////////////////////////////////////////////////////////////////
    tensor_ptr_vec_t Predicter::Predict(const const_tensor_ptr_vec_t& inputs)
    {
        NEURO_ASSERT(inputs.size() == m_InputPlaceholders.size(), "Mismatched number of inputs, expected " << m_InputPlaceholders.size() << " received " << inputs.size() << ".");
        for (size_t i = 0; i < m_InputPlaceholders.size(); ++i)
            m_Feeds[m_InputPlaceholders[i]] = inputs[i];

        return Session::Default()->RunInOrder(m_Order, m_OutputOps, m_Feeds, false);
    }

    //////////////////////////////////////////////////////////////////////////
    tensor_ptr_vec_t Predicter::Eval(const map<Placeholder*, const Tensor*>& feeds)
    {
        return Session::Default()->RunInOrder(m_Order, m_OutputOps, feeds, false);
    }
}