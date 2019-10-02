#include "ComputationalGraph/TensorLike.h"
#include "ComputationalGraph/NameScope.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    TensorLike::TensorLike(const string& name)
    {
        m_Name = NameScope::Name() + name;
    }
}
