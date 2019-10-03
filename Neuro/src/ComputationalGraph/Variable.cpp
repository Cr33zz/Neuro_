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
        Graph::Default()->m_Variables.push_back(this);
    }

    //////////////////////////////////////////////////////////////////////////
    Variable::Variable(float initValue, const string& name)
        : Variable(Tensor({ initValue }, Shape(1)), name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    Variable::Variable(const Shape& shape, InitializerBase* initializer, const string& name)
        : TensorLike(name), m_Initializer(initializer)
    {
        m_Output.Resize(shape);
    }

    //////////////////////////////////////////////////////////////////////////
    void Variable::Init()
    {
        if (m_Initializer)
            m_Initializer->Init(m_Output);
    }
}
