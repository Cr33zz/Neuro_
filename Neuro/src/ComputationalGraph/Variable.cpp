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
        m_Output.SetStorageType(ST_KeepDevMem);
        m_OutputGrad.SetStorageType(ST_Offloadable);
        initValue.CopyTo(m_Output);
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
        m_Output.SetStorageType(ST_KeepDevMem);
        m_OutputGrad.SetStorageType(ST_Offloadable);
        Graph::Default()->AddVariable(this);
    }

    //////////////////////////////////////////////////////////////////////////
    Variable::~Variable()
    {
        delete m_Initializer;
    }

    //////////////////////////////////////////////////////////////////////////
    void Variable::SetTrainable(bool enabled)
    {
        if (m_Trainable == enabled)
            return;

        m_Trainable = enabled;

        for (auto consumer : m_Consumers)
            consumer->RefreshCareAboutGradient();
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
    bool Variable::CareAboutGradient() const
    {
        return m_Trainable || __super::CareAboutGradient();
    }
}
