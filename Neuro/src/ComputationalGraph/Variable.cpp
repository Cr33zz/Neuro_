#include "ComputationalGraph/Variable.h"
#include "ComputationalGraph/Graph.h"
#include "Initializers/InitializerBase.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Variable::Variable(const Tensor& initValue, const string& name)
        : TensorLike(name)
    {
        m_Output.Resize(initValue.GetShape());
        initValue.CopyTo(m_Output);
        m_Output.SetOffloadMode(Offload_KeepAllocated);
        Graph::Default()->AddVariable(this);
        m_Initialized = true;
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
        m_Output.SetOffloadMode(Offload_KeepAllocated);
        Graph::Default()->AddVariable(this);
    }

    //////////////////////////////////////////////////////////////////////////
    Variable::~Variable()
    {
        delete m_Initializer;
    }

    //////////////////////////////////////////////////////////////////////////
    void Variable::Initialize()
    {
        if (m_Initialized)
            return;

        m_Initialized = true;

        if (m_Initializer)
            m_Initializer->Init(m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void Variable::SetOffloadMode(EOffloadMode mode)
    {
        m_Output.SetOffloadMode(mode);
        m_OutputGrad.SetOffloadMode(mode);
    }

}
