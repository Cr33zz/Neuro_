#include "CompGraph/Variable.h"
#include "CompGraph/Graph.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Variable::Variable(const Tensor& initValue)
    {
        m_Value = initValue;
        Graph::Default()->Variables.push_back(this);
    }
}
