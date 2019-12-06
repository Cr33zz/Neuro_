#include "ComputationalGraph/TensorLike.h"
#include "ComputationalGraph/NameScope.h"
#include "ComputationalGraph/Graph.h"

namespace Neuro
{

    //////////////////////////////////////////////////////////////////////////
    Neuro::Tensor& TensorLike::Output()
    {
        //NEURO_ASSERT(!IsConst(), "");
        return m_Output;
    }

    //////////////////////////////////////////////////////////////////////////
    Neuro::Tensor* TensorLike::OutputPtr()
    {
        //NEURO_ASSERT(!IsConst(), "");
        return &m_Output;
    }

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
        : m_UndeterminedOutputShape(false), m_AlwaysOffload(false), m_Fetched(false)
    {
        m_Name = NameScope::Name() + name;
        m_Graph = Graph::Default();
        m_Output.Name(m_Name + "/output");
        m_OutputGrad.Name(m_Name + "/output[grad]");
    }
}
