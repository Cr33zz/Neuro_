#include "ComputationalGraph/Variable.h"
#include "ComputationalGraph/Graph.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Variable::Variable(const Tensor& initValue)
    {
        m_Value = initValue;
        Graph::Default()->Variables.push_back(this);
    }
}
