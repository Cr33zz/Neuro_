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

    class NEURO_DLL_EXPORT Trainer
    {
    public:
        Trainer(const vector<Placeholder*>& inputPlaceholders, const vector<Placeholder*>& targetPlaceholders, const vector<TensorLike*>& fetchOps);

        tensor_ptr_vec_t Train(const const_tensor_ptr_vec_t& inputs, const const_tensor_ptr_vec_t& outputs);

    private:
        vector<Placeholder*> m_InputPlaceholders;
        vector<Placeholder*> m_TargetPlaceholders;
        vector<TensorLike*> m_FetchOps;
        map<Placeholder*, const Tensor*> m_Feeds;

        vector<TensorLike*> m_Order;
    };
}

#pragma warning(pop)