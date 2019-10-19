#include "ComputationalGraph/Operations/FunctionOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    FunctionOp::FunctionOp(const vector<TensorLike*>& inputs, const vector<TensorLike*>& outputs, const string& name)
        : Operation(inputs, name.empty() ? "function" : name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void FunctionOp::ComputeInternal()
    {

    }

    //////////////////////////////////////////////////////////////////////////
    void FunctionOp::ComputeGradientInternal(const Tensor& grad)
    {

    }
}