#include <map>

#include "ComputationalGraph/Predicter.h"
#include "ComputationalGraph/Session.h"
#include "ComputationalGraph/Graph.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Predicter::Predicter(const vector<Placeholder*>& inputPlaceholders, const vector<TensorLike*>& outputOps, Placeholder* trainingPlaceholder)
    {
        m_TrainingPlaceholder = trainingPlaceholder;
        m_InputPlaceholders = inputPlaceholders;
        m_OutputOps = outputOps;

        m_Order = Graph::Default()->BuildForwardOrder(m_OutputOps);

        for (size_t i = 0; i < m_InputPlaceholders.size(); ++i)
            m_Feeds[m_InputPlaceholders[i]] = nullptr;
        
        static Tensor trainingDisabled({ 0.f }, Shape(1));
        m_Feeds[m_TrainingPlaceholder] = &trainingDisabled;
    }

    //////////////////////////////////////////////////////////////////////////
    tensor_ptr_vec_t Predicter::Predict(const const_tensor_ptr_vec_t& inputs)
    {
        for (size_t i = 0; i < m_InputPlaceholders.size(); ++i)
            m_Feeds[m_InputPlaceholders[i]] = inputs[i];

        return Session::Default()->RunInOrder(m_Order, m_OutputOps, m_Feeds);
    }

    //////////////////////////////////////////////////////////////////////////
    tensor_ptr_vec_t Predicter::Eval(const map<Placeholder*, const Tensor*>& feeds)
    {
        auto localFeeds = feeds;
        localFeeds[m_TrainingPlaceholder] = m_Feeds[m_TrainingPlaceholder];
        return Session::Default()->RunInOrder(m_Order, m_OutputOps, localFeeds);
    }
}