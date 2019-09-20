#include "Layers/BatchNormalization.h"
#include "Tools.h"

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
        m_Gamma.CopyTo(targetBatchNorm.m_Gamma, tau);
        m_Beta.CopyTo(targetBatchNorm.m_Beta, tau);
        m_RunningMean.CopyTo(targetBatchNorm.m_RunningMean, tau);
        m_RunningVar.CopyTo(targetBatchNorm.m_RunningVar, tau);
    }

    //////////////////////////////////////////////////////////////////////////
    uint32_t BatchNormalization::ParamsNum() const
    {
        return InputShape().Length * 4;
    }

    //////////////////////////////////////////////////////////////////////////
    void BatchNormalization::GetParametersAndGradients(vector<ParametersAndGradients>& paramsAndGrads, bool onlyTrainable)
    {
        if (onlyTrainable && !m_Trainable)
            return;

        paramsAndGrads.push_back(ParametersAndGradients(&m_Gamma, &m_GammaGrad));
        paramsAndGrads.push_back(ParametersAndGradients(&m_Beta, &m_BetaGrad));
        if (!onlyTrainable)
        {
            paramsAndGrads.push_back(ParametersAndGradients(&m_RunningMean, nullptr));
            paramsAndGrads.push_back(ParametersAndGradients(&m_RunningVar, nullptr));
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
    void BatchNormalization::OnInit()
    {
        __super::OnInit();

        Shape paramsShape = Shape(InputShape().Width(), InputShape().Height(), InputShape().Depth(), 1); // PerActivation
        if (InputShape().Depth() > 1) 
            paramsShape = Shape(1, 1, InputShape().Depth(), 1); // Spatial

        m_Gamma = Tensor(paramsShape, Name() + "/gamma");
        m_Gamma.FillWithValue(1);
        m_Beta = Tensor(paramsShape, Name() + "/beta");
        m_Beta.Zero();

        m_GammaGrad = Tensor(paramsShape, Name() + "/gamma_grad");
        m_GammaGrad.Zero();
        m_BetaGrad = Tensor(paramsShape, Name() + "/beta_grad");
        m_BetaGrad.Zero();

        m_RunningMean = Tensor(paramsShape, Name() + "/running_mean");
        m_RunningMean.FillWithValue(0);
        m_RunningVar = Tensor(paramsShape, Name() + "/running_var");
        m_RunningVar.FillWithValue(1);

        m_SaveMean = Tensor(paramsShape);
        m_SaveMean.Zero();
        m_SaveVariance = Tensor(paramsShape);
        m_SaveVariance.FillWithValue(1);
    }

    //////////////////////////////////////////////////////////////////////////
    void BatchNormalization::OnLink(LayerBase* layer, bool input)
    {
        __super::OnLink(layer, input);

        if (input)
            m_OutputShapes[0] = m_InputShapes[0];
    }

    //////////////////////////////////////////////////////////////////////////
    void BatchNormalization::FeedForwardInternal(bool training)
    {
        if (training)
            m_Inputs[0]->BatchNormalizationTrain(m_Gamma, m_Beta, m_Momentum, m_Epsilon, m_RunningMean, m_RunningVar, m_SaveMean, m_SaveVariance, m_Outputs[0]);
        else
            m_Inputs[0]->BatchNormalization(m_Gamma, m_Beta, m_Epsilon, m_RunningMean, m_RunningVar, m_Outputs[0]);
    }

    //////////////////////////////////////////////////////////////////////////
    void BatchNormalization::BackPropInternal(vector<Tensor>& outputsGradient)
    {
        outputsGradient[0].BatchNormalizationGradient(*m_Inputs[0], m_Gamma, m_Epsilon, outputsGradient[0], m_SaveMean, m_SaveVariance, m_GammaGrad, m_BetaGrad, m_Trainable, m_InputsGradient[0]);
    }
}
