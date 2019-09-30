#include "ComputationalGraph/Placeholder.h"
#include "ComputationalGraph/Graph.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Placeholder::Placeholder(const Shape& shape)
    {
        m_Shape = shape;
        Graph::Default()->Placeholders.push_back(this);
    }
}
