#include <map>

#include "ComputationalGraph/Trainer.h"
#include "ComputationalGraph/Session.h"
#include "ComputationalGraph/Graph.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Trainer::Trainer(const vector<Placeholder*>& inputPlaceholders, const vector<Placeholder*>& targetPlaceholders, const vector<TensorLike*>& fetchOps, Placeholder* trainingPlaceholder)
    {
        m_InputPlaceholders = inputPlaceholders;
        m_TrainingPlaceholder = trainingPlaceholder;
        m_TargetPlaceholders = targetPlaceholders;
        m_FetchOps = fetchOps;

        m_Order = Graph::Default()->BuildForwardOrder(m_FetchOps);

        for (size_t i = 0; i < m_InputPlaceholders.size(); ++i)
            m_Feeds[m_InputPlaceholders[i]] = nullptr;
        for (size_t i = 0; i < m_TargetPlaceholders.size(); ++i)
            m_Feeds[m_TargetPlaceholders[i]] = nullptr;

        static Tensor trainingEnabled({ 1.f }, Shape(1));
        m_Feeds[m_TrainingPlaceholder] = &trainingEnabled;
    }

    //////////////////////////////////////////////////////////////////////////
    tensor_ptr_vec_t Trainer::Train(const const_tensor_ptr_vec_t& inputs, const const_tensor_ptr_vec_t& outputs)
    {
        for (size_t i = 0; i < m_InputPlaceholders.size(); ++i)
            m_Feeds[m_InputPlaceholders[i]] = inputs[i];
        for (size_t i = 0; i < m_TargetPlaceholders.size(); ++i)
            m_Feeds[m_TargetPlaceholders[i]] = outputs[i];

        return Session::Default()->RunInOrder(m_Order, m_FetchOps, m_Feeds);
    }
}