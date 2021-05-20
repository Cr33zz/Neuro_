#pragma once

#include <vector>
#include "Types.h"

#pragma warning(push)
#pragma warning(disable:4251)

namespace Neuro
{
    using namespace std;

    class TensorLike;
    class Placeholder;

    class NEURO_DLL_EXPORT Predicter
    {
    public:
        Predicter(const vector<Placeholder*>& inputPlaceholders, const vector<TensorLike*>& outputOps);

        tensor_ptr_vec_t Predict(const const_tensor_ptr_vec_t& inputs);
        tensor_ptr_vec_t Eval(const map<Placeholder*, const Tensor*>& feeds);

    private:
        vector<Placeholder*> m_InputPlaceholders;
        vector<TensorLike*> m_OutputOps;
        map<Placeholder*, const Tensor*> m_Feeds;

        vector<TensorLike*> m_Order;
    };
}

#pragma warning(pop)