#include "Layers/BatchNormalization.h"
#include "Tools.h"

namespace Neuro
{
    //////////////////////////////////////////////////////////////////////////
    BatchNormalization::BatchNormalization(LayerBase* inputLayer, const string& name)
        : LayerBase(__FUNCTION__, inputLayer, inputLayer->OutputShape(), nullptr, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    BatchNormalization::BatchNormalization(const string& name)
        : LayerBase(__FUNCTION__, Shape(), nullptr, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    BatchNormalization::BatchNormalization(const Shape& inputShape, const string& name)
        : LayerBase(__FUNCTION__, inputShape, inputShape, nullptr, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    void BatchNormalization::CopyParametersTo(LayerBase& target, float tau) const
    {
        __super::CopyParametersTo(target, tau);

        auto& targetBatchNorm = static_cast<BatchNormalization&>(target);
        m_Gamma.CopyTo(targetBatchNorm.m_Gamma, tau);
        m_Beta.CopyTo(targetBatchNorm.m_Beta, tau);
    }

    //////////////////////////////////////////////////////////////////////////
    uint32_t BatchNormalization::ParamsNum() const
    {
        return m_Gamma.Length() + m_Beta.Length();
    }

    //////////////////////////////////////////////////////////////////////////
    void BatchNormalization::GetParametersAndGradients(vector<ParametersAndGradients>& paramsAndGrads, bool onlyTrainable)
    {
        if (onlyTrainable && !m_Trainable)
            return;

        paramsAndGrads.push_back(ParametersAndGradients(&m_Gamma, &m_GammaGrad));
        paramsAndGrads.push_back(ParametersAndGradients(&m_Beta, &m_BetaGrad));
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

        m_Gamma = Tensor(Shape(m_OutputShapes[0].Width(), m_OutputShapes[0].Height(), m_OutputShapes[0].Depth()), Name() + "/gamma");
        m_Beta = Tensor(m_Gamma.GetShape(), Name() + "/beta");
        m_RunningMean = Tensor(m_Gamma.GetShape());
        m_RunningVar = Tensor(m_Gamma.GetShape());

        m_GammaGrad = Tensor(m_Gamma.GetShape(), Name() + "/gamma_grad");
        m_BetaGrad = Tensor(m_Beta.GetShape(), Name() + "/beta_grad");

        m_Gamma.FillWithValue(1);
        m_Beta.FillWithValue(0);
        m_RunningMean.FillWithValue(0);
        m_RunningVar.FillWithValue(1);
    }

    //////////////////////////////////////////////////////////////////////////
    void BatchNormalization::OnLink()
    {
        m_OutputShapes[0] = m_InputShapes[0];
    }

    //////////////////////////////////////////////////////////////////////////
    void BatchNormalization::FeedForwardInternal(bool training)
    {
        if (training)
            m_Inputs[0]->BatchNormalizationTrain(m_Gamma, m_Beta, m_Momentum, m_RunningMean, m_RunningVar, m_Mean, m_Variance, m_Outputs[0]);
        else
            m_Inputs[0]->BatchNormalization(m_Gamma, m_Beta, m_RunningMean, m_RunningVar, m_Outputs[0]);
    }

    //////////////////////////////////////////////////////////////////////////
    void BatchNormalization::BackPropInternal(vector<Tensor>& outputsGradient)
    {
        outputsGradient[0].BatchNormalizationGradient(*m_Inputs[0], m_Gamma, outputsGradient[0], m_Mean, m_Variance, m_GammaGrad, m_BetaGrad, m_InputsGradient[0]);
    }
}
