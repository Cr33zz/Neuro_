#include "ComputationalGraph/Placeholder.h"
#include "ComputationalGraph/Graph.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Placeholder::Placeholder(const Shape& shape, const string& name)
        : TensorLike(name)
    {
        m_Output.Resize(shape);
        m_Output.Zero();
        Graph::Default()->AddPlaceholder(this);
    }

    //////////////////////////////////////////////////////////////////////////
    Placeholder::Placeholder(const Tensor& defaultVal, const string& name)
        : TensorLike(name)
    {
        m_Output.Resize(defaultVal.GetShape());
        defaultVal.CopyTo(m_Output);
        Graph::Default()->AddPlaceholder(this);
    }
}
