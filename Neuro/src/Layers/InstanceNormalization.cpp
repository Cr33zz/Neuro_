#include "Layers/InstanceNormalization.h"
#include "ComputationalGraph/Variable.h"
#include "ComputationalGraph/Ops.h"

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

        m_Built = true;
    }

    //////////////////////////////////////////////////////////////////////////
    vector<TensorLike*> InstanceNormalization::InternalCall(const vector<TensorLike*>& inputs, TensorLike* training)
    {
        auto _mean = mean(inputs[0], _01Axes);
        auto xmu = sub(inputs[0], _mean);
        auto var = mean(square(xmu), _01Axes, "variance");
        auto normed = divide(xmu, add(sqrt(var, "stddev"), m_Epsilon), "normed");
        return { add(multiply(normed, m_Gamma, "gamma_scaled"), m_Beta, "beta_shifted") };
    }
}