#include "ComputationalGraph/Placeholder.h"
#include "ComputationalGraph/Graph.h"
#include "ComputationalGraph/NameScope.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    Placeholder::Placeholder(const Shape& shape, const string& name)
    {
        m_Shape = shape;
        m_Name = NameScope::Name() + name;
        //Graph::Default()->Placeholders.push_back(this);
    }
}
