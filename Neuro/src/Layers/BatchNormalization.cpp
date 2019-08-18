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
    BatchNormalization::BatchNormalization(const Shape& inputShape, const string& name)
        : LayerBase(__FUNCTION__, inputShape, inputShape, nullptr, name)
    {
    }

    //////////////////////////////////////////////////////////////////////////
    BatchNormalization::BatchNormalization()
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
    int BatchNormalization::GetParamsNum() const
    {
        return m_Gamma.Length() + m_Beta.Length();
    }

    //////////////////////////////////////////////////////////////////////////
    void BatchNormalization::GetParametersAndGradients(vector<ParametersAndGradients>& result)
    {
        result.push_back(ParametersAndGradients(&m_Gamma, &m_GammaGrad));
        result.push_back(ParametersAndGradients(&m_Beta, &m_BetaGrad));
    }

    //////////////////////////////////////////////////////////////////////////
    LayerBase* BatchNormalization::GetCloneInstance() const
    {
        return new BatchNormalization();
    }

    //////////////////////////////////////////////////////////////////////////
    void BatchNormalization::OnInit()
    {
        __super::OnInit();

        m_Gamma = Tensor(Shape(m_OutputShape.Width(), m_OutputShape.Height(), m_OutputShape.Depth()));
        m_Beta = Tensor(m_Gamma.GetShape());
        m_RunningMean = Tensor(m_Gamma.GetShape());
        m_RunningVar = Tensor(m_Gamma.GetShape());

        m_GammaGrad = Tensor(m_Gamma.GetShape());
        m_BetaGrad = Tensor(m_Beta.GetShape());

        m_Gamma.FillWithValue(1);
        m_Beta.FillWithValue(0);
        m_RunningMean.FillWithValue(1);
        m_RunningVar.FillWithValue(1);
    }

    //////////////////////////////////////////////////////////////////////////
    void BatchNormalization::FeedForwardInternal(bool training)
    {
        if (training)
        {
            float N = (float)m_Inputs[0]->Batch();
            mu = m_Inputs[0]->SumBatches().Mul(1.f / N);
            xmu = m_Inputs[0]->Sub(mu);
            carre = xmu.Map([](float x) { return x * x; });
            var = carre.SumBatches().Mul(1.f / N);
            sqrtvar = var.Map([](float x) { return sqrt(x); });
            invvar = sqrtvar.Map([](float x) { return 1.f / x; });
            va2 = xmu.MulElem(invvar);
            va2.Map([&](float x) { return x * m_Gamma(0) + m_Beta(0); }, m_Output);

            m_RunningMean.Map([&](float x1, float x2) { return m_Momentum * x1 + (1.f - m_Momentum) * x2; }, mu, m_RunningMean);
            m_RunningVar.Map([&](float x1, float x2) { return m_Momentum * x1 + (1.f - m_Momentum) * x2; }, var, m_RunningVar);
        }
        else
        {
            Tensor xbar = m_Inputs[0]->Sub(m_RunningMean);
            xbar.Map([&](float x1, float x2) { return x1 / sqrt(x2 + _EPSILON); }, m_RunningVar, xbar);
            xbar.Map([&](float x) { return x * m_Gamma(0) + m_Beta(0); }, m_Output);
        }
    }

    //////////////////////////////////////////////////////////////////////////
    void BatchNormalization::BackPropInternal(Tensor& outputGradient)
    {
        float N = (float)outputGradient.Batch();

        m_BetaGrad = outputGradient.SumBatches();
        m_GammaGrad = va2.MulElem(outputGradient).SumBatches();
        
        Tensor dva2 = outputGradient.MulElem(m_Gamma);
        Tensor dxmu = dva2.MulElem(invvar);
        Tensor dinvvar = xmu.MulElem(dva2).SumBatches();
        Tensor dsqrtvar = dinvvar.Map([&](float x1, float x2) { return -1.f / (x2*x2) * x1; }, sqrtvar);
        Tensor dvar = dsqrtvar.Map([&](float x1, float x2) { return 0.5f * pow(x2 + _EPSILON, -0.5f) * x1; }, var);
        Tensor dcarre = Tensor(carre.GetShape()).FillWithValue(1).MulElem(dvar).Mul(1.f / N);
        dxmu.Add(xmu.MulElem(dcarre).Mul(2), dxmu);
        Tensor dmu = dxmu.SumBatches().Negated();
        dxmu.Add(Tensor(dxmu.GetShape()).FillWithValue(1).MulElem(dmu).Mul(1.f / N), m_InputsGradient[0]);
    }
}
