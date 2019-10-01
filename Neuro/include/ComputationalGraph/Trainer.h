#pragma once

#include <vector>
#include "Types.h"

namespace Neuro
{
    using namespace std;

    class NodeBase;
    class Placeholder;

    class Trainer
    {
    public:
        Trainer(const vector<Placeholder*>& inputOps, const vector<Placeholder*>& targetOps, const vector<NodeBase*>& updatesOps);

        tensor_ptr_vec_t Train(const const_tensor_ptr_vec_t& inputs, const const_tensor_ptr_vec_t& outputs);

    private:
        vector<Placeholder*> m_InputOps;
        vector<Placeholder*> m_TargetOps;
        vector<NodeBase*> m_UpdateOps;
    };
}
