#pragma once

#include <vector>
#include "Types.h"

namespace Neuro
{
    using namespace std;

    class TensorLike;
    class Placeholder;

    class Predicter
    {
    public:
        Predicter(const vector<Placeholder*>& inputOps, const vector<TensorLike*>& outputOps, Placeholder* trainingPlaceholder);

        tensor_ptr_vec_t Predict(const const_tensor_ptr_vec_t& inputs);

    private:
        vector<Placeholder*> m_InputPlaceholders;
        Placeholder* m_TrainingPlaceholder;
        vector<TensorLike*> m_OutputOps;
        map<Placeholder*, const Tensor*> m_Feeds;

        vector<TensorLike*> m_Order;
    };
}
