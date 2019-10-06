#include <map>

#include "ComputationalGraph/Predicter.h"
#include "ComputationalGraph/Session.h"
#include "Tensors/Tensor.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Predicter::Predicter(const vector<Placeholder*>& inputPlaceholders, const vector<TensorLike*>& outputOps, Placeholder* trainingPlaceholder)
    {
        m_TrainingPlaceholder = trainingPlaceholder;
        m_InputPlaceholders = inputPlaceholders;
        m_OutputOps = outputOps;


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

        return Session::Default()->Run(m_OutputOps, m_Feeds);
    }
}