#include "ComputationalGraph/Operations/DumpOp.h"
#include "Models/ModelBase.h"
#include "Tools.h"
#include "Debug.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    DumpOp::DumpOp(TensorLike* x, const string& name)
        : Operation({ x }, name.empty() ? x->Name() : name)
    {
        UpdateOutputShape();
    }

    //////////////////////////////////////////////////////////////////////////
    void DumpOp::ComputeInternal()
    {
        m_Inputs[0]->DebugDumpValues(Name() + "_step" + to_string(Debug::GetStep()) + ".log");
        m_Output.ResizeBatch(m_Inputs[0]->Batch());
        m_Inputs[0]->CopyTo(m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void DumpOp::ComputeGradientInternal(const Tensor& grad)
    {   
        grad.CopyTo(m_InputsGrads[0]);
    }
}