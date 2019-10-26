#include "Layers/BatchNormalization.h"
#include "Activations.h"
#include "ComputationalGraph/Variable.h"
#include "ComputationalGraph/Ops.h"
#include "Tools.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    BatchNormalization::BatchNormalization(const string& name)
        : SingleLayer(__FUNCTION__, Shape(), nullptr, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    BatchNormalization::BatchNormalization(const Shape& inputShape, const string& name)
        : SingleLayer(__FUNCTION__, inputShape, nullptr, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    BatchNormalization::BatchNormalization(const string& constructorName, const Shape& inputShape, const string& name)
        : SingleLayer(constructorName, inputShape, nullptr, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void BatchNormalization::CopyParametersTo(LayerBase& target, float tau) const
    {
        __super::CopyParametersTo(target, tau);

        auto& targetBatchNorm = static_cast<BatchNormalization&>(target);
        m_Gamma->Output().CopyTo(targetBatchNorm.m_Gamma->Output(), tau);
        m_Beta->Output().CopyTo(targetBatchNorm.m_Beta->Output(), tau);
        m_RunningMean->Output().CopyTo(targetBatchNorm.m_RunningMean->Output(), tau);
        m_RunningVar->Output().CopyTo(targetBatchNorm.m_RunningVar->Output(), tau);
    }

    //////////////////////////////////////////////////////////////////////////
    void BatchNormalization::Parameters(vector<Variable*>& params, bool onlyTrainable) const
    {
        if (onlyTrainable && !m_Trainable)
            return;

        params.push_back(m_Gamma);
        params.push_back(m_Beta);
        if (!onlyTrainable)
        {
            params.push_back(m_RunningMean);
            params.push_back(m_RunningVar);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    Neuro::BatchNormalization* BatchNormalization::SetMomentum(float momentum)
    {
        m_Momentum = momentum;
        return this;
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* BatchNormalization::GetCloneInstance() const
    {
        return new BatchNormalization(false);
    }

    //////////////////////////////////////////////////////////////////////////
    void BatchNormalization::Build(const vector<Shape>& inputShapes)
    {
        NEURO_ASSERT(inputShapes.size() == 1, "Dense layer accepts single input.");

        Shape paramsShape = Shape(inputShapes[0].Width(), inputShapes[0].Height(), inputShapes[0].Depth(), 1); // PerActivation
        if (inputShapes[0].Depth() > 1)
            paramsShape = Shape(1, 1, inputShapes[0].Depth(), 1); // Spatial

        m_Gamma = new Variable(ones(paramsShape), "gamma");
        m_Beta = new Variable(zeros(paramsShape), "beta");

        m_RunningMean = new Variable(zeros(paramsShape), "running_mean");
        m_RunningMean->SetTrainable(false);
        m_RunningVar = new Variable(ones(paramsShape), "running_var");
        m_RunningVar->SetTrainable(false);

        m_Built = true;
    }

    //////////////////////////////////////////////////////////////////////////
    vector<TensorLike*> BatchNormalization::InternalCall(const vector<TensorLike*>& inputs, TensorLike* training)
    {
        TensorLike* output = batch_norm(inputs[0], m_Gamma, m_Beta, m_RunningMean, m_RunningVar, m_Momentum, m_Epsilon, training);
        if (m_Activation)
            output = m_Activation->Build(output);

        return { output };
    }
}
