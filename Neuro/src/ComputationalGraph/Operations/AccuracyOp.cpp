#include "ComputationalGraph/Operations/AccuracyOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    AccuracyOp::AccuracyOp(TensorLike* target, TensorLike* output, const string& name)
        : Operation({ target, output }, name.empty() ? "accuracy" : name)
    {
        m_Output.Resize(Shape(1));
    }

    //////////////////////////////////////////////////////////////////////////
    void AccuracyOp::ComputeInternal()
    {
        auto& target = *m_Inputs[0];
        auto& output = *m_Inputs[1];

        Tensor targetArgMax = target.ArgMax(EAxis::_012Axes);
        targetArgMax.Reshape(Shape(target.Batch()));
        Tensor outputArgMax = output.ArgMax(EAxis::_012Axes);
        outputArgMax.Reshape(Shape(output.Batch()));

        int hits = 0;
        for (uint32_t i = 0; i < targetArgMax.Length(); ++i)
            hits += targetArgMax(i) == outputArgMax(i) ? 1 : 0;

        m_Output.SetFlat((float)hits / output.Batch(), 0);
    }

    //////////////////////////////////////////////////////////////////////////
    BinaryAccuracyOp::BinaryAccuracyOp(TensorLike* target, TensorLike* output, const string& name)
        : Operation({ target, output }, name.empty() ? "binary_accuracy" : name)
    {
        m_Output.Resize(Shape(1));
    }

    //////////////////////////////////////////////////////////////////////////
    void BinaryAccuracyOp::ComputeInternal()
    {
        auto& target = *m_Inputs[0];
        auto& output = *m_Inputs[1];

        int hits = 0;
        for (uint32_t n = 0; n < output.Batch(); ++n)
            hits += target.GetFlat(n) == roundf(output.GetFlat(n)) ? 1 : 0;

        m_Output.SetFlat((float)hits / output.Batch(), 0);
    }
}