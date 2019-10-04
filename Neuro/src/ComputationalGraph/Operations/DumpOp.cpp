#include "ComputationalGraph/Operations/DumpOp.h"
#include "Models/ModelBase.h"
#include "Tools.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    DumpOp::DumpOp(TensorLike* x, const string& name)
        : Operation({ x }, name.empty() ? x->Name() : name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void DumpOp::ComputeInternal()
    {
        m_Inputs[0]->DebugDumpValues(Replace(Name() + "_step" + to_string(g_LogOutputsStep) + ".log", "/", "_"));
        m_Output.Resize(m_Inputs[0]->GetShape());
        m_Inputs[0]->CopyTo(m_Output);
    }

    //////////////////////////////////////////////////////////////////////////
    void DumpOp::ComputeGradientInternal(const Tensor& grad)
    {   
        grad.CopyTo(m_InputsGrads[0]);
    }
}