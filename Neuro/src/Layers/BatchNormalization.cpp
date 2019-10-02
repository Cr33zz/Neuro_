#include "Layers/BatchNormalization.h"
#include "Tools.h"
#include "ComputationalGraph/Variable.h"
#include "ComputationalGraph/Ops.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    BatchNormalization::BatchNormalization(LayerBase* inputLayer, const string& name)
        : SingleLayer(__FUNCTION__, inputLayer, inputLayer->OutputShape(), nullptr, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    BatchNormalization::BatchNormalization(const string& name)
        : SingleLayer(__FUNCTION__, Shape(), nullptr, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    BatchNormalization::BatchNormalization(const Shape& inputShape, const string& name)
        : SingleLayer(__FUNCTION__, inputShape, inputShape, nullptr, name)
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
    uint32_t BatchNormalization::ParamsNum() const
    {
        return (InputShape().Depth() > 1 ? InputShape().Depth() : InputShape().Length) * 4;
    }

    //////////////////////////////////////////////////////////////////////////
    void BatchNormalization::ParametersAndGradients(vector<ParameterAndGradient>& paramsAndGrads, bool onlyTrainable)
    {
        if (onlyTrainable && !m_Trainable)
            return;

        paramsAndGrads.push_back(ParameterAndGradient(&m_Gamma->Output()));
        paramsAndGrads.push_back(ParameterAndGradient(&m_Beta->Output()));
        if (!onlyTrainable)
        {
            paramsAndGrads.push_back(ParameterAndGradient(&m_RunningMean->Output()));
            paramsAndGrads.push_back(ParameterAndGradient(&m_RunningVar->Output()));
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
    void BatchNormalization::InitOps(TensorLike* training, bool initValues)
    {
        Shape paramsShape = Shape(InputShape().Width(), InputShape().Height(), InputShape().Depth(), 1); // PerActivation
        if (InputShape().Depth() > 1) 
            paramsShape = Shape(1, 1, InputShape().Depth(), 1); // Spatial

        m_Gamma = new Variable(ones(paramsShape), "gamma");
        m_Beta = new Variable(zeros(paramsShape), "beta");

        m_RunningMean = new Variable(zeros(paramsShape), "running_mean");
        m_RunningVar = new Variable(ones(paramsShape), "running_var");

        m_OutputOps[0] = batch_norm(m_InputOps[0], m_Gamma, m_Beta, m_RunningMean, m_RunningVar, m_Momentum, m_Epsilon, training);
    }

    //////////////////////////////////////////////////////////////////////////
    void BatchNormalization::OnLinkInput(const vector<LayerBase*>& inputLayers)
    {
        __super::OnLinkInput(inputLayers);

        m_OutputsShapes[0] = m_InputShape;
    }
}
