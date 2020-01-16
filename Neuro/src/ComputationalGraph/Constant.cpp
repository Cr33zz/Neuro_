#include "ComputationalGraph/Constant.h"
#include "ComputationalGraph/Graph.h"
#include "ComputationalGraph/NameScope.h"

namespace Neuro
{
    int Constant::s_NameId = 0;

    //////////////////////////////////////////////////////////////////////////
    Constant::Constant(const Tensor& value, const string& name)
        : TensorLike(name == "" ? ("const_" + to_string(++s_NameId)) : name)
    {
        m_Output.Resize(value.GetShape());
        m_Output.SetStorageType(ST_KeepDevMem);
        value.CopyTo(m_Output);
        Graph::Default()->AddConstant(this);
    }

    //////////////////////////////////////////////////////////////////////////
    Constant::Constant(float value, const string& name)
        : Constant(Tensor({ value }, Shape(1)), name)
    {
    }
}
