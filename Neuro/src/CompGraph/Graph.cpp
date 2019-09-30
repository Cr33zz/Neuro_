#include "CompGraph/Graph.h"

namespace Neuro
{
    Graph* Graph::s_Default = nullptr;

    //////////////////////////////////////////////////////////////////////////
    Graph::Graph()
    {
        if (!s_Default)
            SetAsDefault();
    }
}
