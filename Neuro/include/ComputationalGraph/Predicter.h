#pragma once

#include <vector>
#include "Types.h"

namespace Neuro
{
    using namespace std;

    class NodeBase;
    class Placeholder;

    class Predicter
    {
    public:
        Predicter(const vector<Placeholder*>& inputOps, const vector<NodeBase*>& outputOps);

        tensor_ptr_vec_t Predict(const const_tensor_ptr_vec_t& inputs);

    private:
        vector<Placeholder*> m_InputOps;
        vector<NodeBase*> m_OutputOps;
    };
}
