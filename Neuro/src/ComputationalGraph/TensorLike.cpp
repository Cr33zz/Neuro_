#include "ComputationalGraph/TensorLike.h"
#include "ComputationalGraph/NameScope.h"
#include "ComputationalGraph/Graph.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    void TensorLike::Preload()
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void TensorLike::PreloadForGradient()
    {
        Output().Prefetch();
        for (auto inputNode : m_InputNodes)
            inputNode->Output().Prefetch();
    }

    //////////////////////////////////////////////////////////////////////////
    bool TensorLike::CareAboutGradient() const
    {
        return !m_InputNodes.empty();
    }

    //////////////////////////////////////////////////////////////////////////
    TensorLike::TensorLike(const string& name)
        : m_AlwaysOffload(false), m_Fetched(false)
    {
        m_Name = NameScope::Name() + name;
        m_Graph = Graph::Default();
        m_Output.Name(m_Name + "/output");
        m_OutputGrad.Name(m_Name + "/output[grad]");
    }
}
