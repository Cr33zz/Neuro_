#pragma once

#include <string>

#include "Tensors/Tensor.h"

namespace Neuro
{
    using namespace std;

    class NodeBase
    {
    public:
        const string& Name() const { return m_Name; }

    private:
        vector<NodeBase*> m_Consumers;
        vector<NodeBase*> m_InputNodes;
        Tensor m_Output;
        string m_Name;
    };
}
