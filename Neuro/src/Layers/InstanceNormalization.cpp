#include "Layers/InstanceNormalization.h"
#include "ComputationalGraph/Variable.h"
#include "ComputationalGraph/Operations/InstanceNormalizeOp.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    InstanceNormalization::InstanceNormalization(const string& name)
        : BatchNormalization(__FUNCTION__, Shape(), name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    InstanceNormalization::InstanceNormalization(const Shape& inputShape, const string& name)
        : BatchNormalization(__FUNCTION__, inputShape, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void InstanceNormalization::Build(const vector<Shape>& inputShapes)
    {
        NEURO_ASSERT(inputShapes.size() == 1, "Dense layer accepts single input.");

        Shape paramsShape = Shape(1, 1, inputShapes[0].Depth(), inputShapes[0].Batch());

        m_Gamma = new Variable(ones(paramsShape), "gamma");
        m_Beta = new Variable(zeros(paramsShape), "beta");

        m_RunningMean = new Variable(zeros(paramsShape), "running_mean");
        m_RunningVar = new Variable(ones(paramsShape), "running_var");

        m_Built = true;
    }

    //////////////////////////////////////////////////////////////////////////
    vector<TensorLike*> InstanceNormalization::InternalCall(const vector<TensorLike*>& inputs, TensorLike* training)
    {
        TensorLike* output = instance_norm(inputs[0], m_Gamma, m_Beta, m_RunningMean, m_RunningVar, m_Momentum, m_Epsilon, training);
        return { output };
    }
}