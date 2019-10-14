#include "ComputationalGraph/Constant.h"
#include "ComputationalGraph/Graph.h"
#include "ComputationalGraph/NameScope.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Constant::Constant(const Tensor& value, const string& name)
        : TensorLike(name)
    {
        m_Output = value;
        m_Output.SetOffloadMode(Offload_KeepAllocated);
    }

    //////////////////////////////////////////////////////////////////////////
    Constant::Constant(float value, const string& name)
        : Constant(Tensor({ value }, Shape(1)))
    {
    }
}
