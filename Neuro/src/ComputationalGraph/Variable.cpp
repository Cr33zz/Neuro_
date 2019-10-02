#include "ComputationalGraph/Variable.h"
#include "ComputationalGraph/Graph.h"
#include "Initializers/InitializerBase.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Variable::Variable(const Tensor& initValue, const string& name)
        : TensorLike(name)
    {
        m_Output = initValue;
        //Graph::Default()->Variables.push_back(this);
    }

    //////////////////////////////////////////////////////////////////////////
    Variable::Variable(const Shape& shape, InitializerBase* initializer, const string& name)
        : TensorLike(name)
    {
        m_Output.Resize(shape);
        if (initializer)
            initializer->Init(m_Output);
    }
}
