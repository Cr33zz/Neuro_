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
        //Graph::Default()->Placeholders.push_back(this);
    }
}
